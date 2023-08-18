from tqdm import tqdm   
from biva.utils import summary2logger, save_model, sample_model
from biva.utils import LowerBoundedExponentialLR,training_step, test_step, summary2logger, save_model, load_model, sample_model, DiscretizedMixtureLogits
from load_deepvae import visual_eval
from booster.utils import EMA, logging_sep, available_device
from booster import Aggregator
import torch
from time import time
import wandb
import logging
import os

def append_ellapsed_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        diagnostics = func(*args, **kwargs)
        diagnostics['info']['elapsed-time'] = time() - start_time
        return diagnostics

    return wrapper


@append_ellapsed_time
def training_step(x, model, evaluator, optimizer, scheduler=None, **kwargs):
    optimizer.zero_grad()
    model.train()

    loss, diagnostics, output = evaluator(model, x, **kwargs)
    loss = loss.mean(0)

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return diagnostics


@torch.no_grad()
@append_ellapsed_time
def test_step(x, model, evaluator, **kwargs):
    model.eval()
    loss, diagnostics, output = evaluator(model, x, **kwargs)

    return diagnostics

def train(model, likelihood, ema, train_loader, val_loader, optimizer, scheduler, evaluator, train_aggregator, val_aggregator, EPOCHS, global_step, logdir, **kwargs):
    
    train_logger = logging.getLogger('train')
    eval_logger = logging.getLogger('eval')
    
    best_elbo = (-1e20, 0, 0)

    for epoch in range(1, EPOCHS + 1):

        # training
        train_aggregator.initialize()
        for x in tqdm(train_loader, desc='train epoch'):
            x = x.to('cuda')
            # x = torch.cat((x,)*3, axis=1)
            diagnostics = training_step(x, model, evaluator, optimizer, scheduler, **kwargs)
            train_aggregator.update(diagnostics)
            ema.update()
            global_step += 1
            
        train_summary = train_aggregator.data.to('cpu')
        
        # train_summary = {k: v.item() for k, v in train_summary.get('loss').items()}


        # evaluation
        val_aggregator.initialize()
        for x in tqdm(val_loader, desc='val epoch'):
            x = x.to('cuda')
            # x = torch.cat((x,)*3, axis=1)

            diagnostics = test_step(x, ema.model, evaluator, **kwargs)
            val_aggregator.update(diagnostics)

        eval_summary = val_aggregator.data.to('cpu')

        # keep best model
        best_elbo = save_model(ema.model, eval_summary, global_step, epoch, best_elbo, logdir)

        # print(train_summary)
        # logging
        summary2logger(train_logger, train_summary, global_step, epoch)
        summary2logger(eval_logger, eval_summary, global_step, epoch, best_elbo)

        eval_summary = {k: v.item() for k, v in eval_summary.get('loss').items()}
        train_summary = {k: v.item() for k, v in train_summary.get('loss').items()}

        # wandb logging
        wandb.log({'train':train_summary})
        wandb.log({'val':eval_summary})
        
        visual_eval(x, model, likelihood, logdir)

        # sample model
        sample_model(ema.model, likelihood, logdir, global_step=global_step, N=100)