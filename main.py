import argparse
import json
import logging
import wandb
import os
import pickle

import numpy as np
import torch
from biva.dataloaders import get_dataloaders

from biva.evaluation import VariationalInference
from biva.model import DeepVae, get_deep_vae_mnist, get_deep_vae_cifar, get_deep_vae_brain, get_deep_vae_abdom, VaeStage, LvaeStage, BivaStage
from biva.utils import LowerBoundedExponentialLR, training_step, test_step, summary2logger, save_model, load_model, \
    sample_model, DiscretizedMixtureLogits
from booster import Aggregator
from booster.utils import EMA, logging_sep, available_device
from torch.distributions import Bernoulli
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='runs/', help='directory to store training logs')
parser.add_argument('--likelihood', action='store_false', help='approx. elbo through reconstruction')
parser.add_argument('--data_root', default='data/', help='directory to store the dataset')
parser.add_argument('--dataset', default='binmnist', help='binmnist')
parser.add_argument('--model_type', default='biva', help='model type (vae | lvae | biva)')
parser.add_argument('--device', default='auto', help='auto, cuda, cpu')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--bs', default=48, type=int, help='batch size')
parser.add_argument('--epochs', default=500, type=int, help='number of epochs')
parser.add_argument('--lr', default=2e-3, type=float, help='base learning rate')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--freebits', default=2.0, type=float, help='freebits per latent variable')
parser.add_argument('--nr_mix', default=10, type=int, help='number of mixtures')
parser.add_argument('--ema', default=0.9995, type=float, help='ema')
parser.add_argument('--q_dropout', default=0.5, type=float, help='inference model dropout')
parser.add_argument('--p_dropout', default=0.5, type=float, help='generative model dropout')
parser.add_argument('--iw_samples', default=1000, type=int, help='number of importance weighted samples for testing')
parser.add_argument('--id', default='', type=str, help='run id suffix')
parser.add_argument('--no_skip', action='store_true', help='do not use skip connections')
parser.add_argument('--log_var_act', default='softplus', type=str, help='activation for the log variance')
parser.add_argument('--beta', default=1.0, type=float, help='Beta parameter (Beta-VAE)')

args = parser.parse_args()

# set random seed, set run-id, init log directory and save config
torch.manual_seed(args.seed)
np.random.seed(args.seed)
run_id = f"{args.dataset}-{args.model_type}-seed{args.seed}"
if len(args.id):
    run_id += f"-{args.id}"
if args.beta != 1:
    run_id += f"-{args.beta}"
if args.likelihood == False:
    run_id += 'reconstruction_elbo'
logdir = os.path.join(args.root, run_id)
if not os.path.exists(logdir):
    os.makedirs(logdir)
with open(os.path.join(logdir, 'config.json'), 'w') as fp:
    fp.write(json.dumps(vars(args)))

# define tensorboard writers
wandb.init(project="dirbiva2d", entity="rachaelharkness1", notes=run_id)

train_writer = SummaryWriter(os.path.join(logdir, 'train'))
valid_writer = SummaryWriter(os.path.join(logdir, 'valid'))

# define training and validation data loaders
train_loader, valid_loader, test_loader, tensor_shp = get_dataloaders(args.dataset, args.data_root, args.bs, args.num_workers)

# define likelihood
if args.likelihood:
    likelihood = {'cifar10': DiscretizedMixtureLogits(args.nr_mix), 'binmnist': Bernoulli}[args.dataset]
else:
    likelihood = None

# define model
if 'cifar' in args.dataset:
    stages, latents = get_deep_vae_cifar()
    if likelihood is None:
        features_out = 10 * args.nr_mix
    else:
        features_out = tensor_shp[1]

elif 'mood' in args.dataset:
    if 'brain' in args.dataset:
        stages, latents = get_deep_vae_brain()
        if likelihood is None:
            features_out = tensor_shp[1]
        else:
            features_out = 10 * args.nr_mix

    if 'abdom' in args.dataset:
        stages, latents = get_deep_vae_abdom()
        if likelihood is None:
            features_out = tensor_shp[1]
        else:
            features_out = 10 * args.nr_mix    
else:
    stages, latents = get_deep_vae_mnist()
    features_out = tensor_shp[1]

Stage = {'vae': VaeStage, 'lvae': LvaeStage, 'biva': BivaStage}[args.model_type]
log_var_act = {'none': None, 'softplus': torch.nn.Softplus, 'tanh': torch.nn.Tanh}[args.log_var_act]
hyperparameters = {
    'Stage': Stage,
    'tensor_shp': tensor_shp,
    'stages': stages,
    'latents': latents,
    'nonlinearity': 'elu',
    'q_dropout': args.q_dropout,
    'p_dropout': args.p_dropout,
    'type': args.model_type,
    'features_out': features_out,
    'no_skip': args.no_skip,
    'log_var_act': log_var_act
}
# save hyper parameters for easy loading
pickle.dump(hyperparameters, open(os.path.join(logdir, "hyperparameters.p"), "wb"))

# instantiate the model and move to target device
model = DeepVae(**hyperparameters)
device = available_device() if args.device == 'auto' else args.device
model.to(device)

# define the evaluator
evaluator = VariationalInference(likelihood, iw_samples=1)

# define evaluation model with Exponential Moving Average
ema = EMA(model, args.ema)

# data dependent init for weight normalization (automatically done during the first forward pass)
with torch.no_grad():
    model.train()
    x = next(iter(train_loader)).to(device)
    model(x)

# print stages
print(logging_sep("=") + "\nGenerative model:\n" + logging_sep("-"))
for i, (convs, z) in reversed(list(enumerate(zip(stages, latents)))):
    print(f"Stage #{i + 1}")
    print("Stochastic layer:", z)
    print("Deterministic block:", convs)
print(logging_sep("="))

# define freebits
n_latents = len(latents)
if args.model_type == 'biva':
    n_latents = 2 * n_latents - 1
freebits = [args.freebits] * n_latents

# optimizer
optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999,))
scheduler = LowerBoundedExponentialLR(optimizer, 0.999999, 0.0001)

# logging utils
kwargs = {'beta': args.beta, 'freebits': freebits}
best_elbo = (-1e20, 0, 0)
global_step = 1
train_agg = Aggregator()
val_agg = Aggregator()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.FileHandler(os.path.join(logdir, 'run.log')),
                              logging.StreamHandler()])
train_logger = logging.getLogger('train')
eval_logger = logging.getLogger('eval')
M_parameters = (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
logging.getLogger(run_id).info(f'# Total Number of Parameters: {M_parameters:.3f}M')
print(logging_sep() + f"\nLogging directory: {logdir}\n" + logging_sep())

# init sample
sample_model(ema.model, likelihood, logdir, writer=valid_writer, global_step=global_step, N=100)




# load best model
load_model(ema.model, logdir)

# sample model
sample_model(ema.model, likelihood, logdir, N=100)

# final test
iw_evaluator = VariationalInference(likelihood, iw_samples=args.iw_samples)
test_agg = Aggregator()
test_logger = logging.getLogger('test')
test_logger.info(f"best elbo at step {best_elbo[1]}, epoch {best_elbo[2]}: {best_elbo[0]:.3f} nats")

test_agg.initialize()
for x in tqdm(test_loader, desc='iw test epoch'):
    x = x.to(device)
    diagnostics = test_step(x, ema.model, iw_evaluator, **kwargs)
    test_agg.update(diagnostics)
test_summary = test_agg.data.to('cpu')

summary2logger(test_logger, test_summary, best_elbo[1], best_elbo[2], None)
