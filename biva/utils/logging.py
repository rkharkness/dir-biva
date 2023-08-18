import math
import os

import matplotlib.image
import numpy as np
import torch
from torchvision.utils import make_grid
import wandb
import matplotlib.pyplot as plt
from booster.utils import logging_sep


def build_and_save_grid(data, logdir, filename, N=100):
    nrow = math.floor(math.sqrt(N))
    grid = make_grid(data, nrow=nrow)

    # normalize
    grid -= grid.min()
    grid /= grid.max()

    # save the raw image
    img = grid.data.permute(1, 2, 0).cpu().numpy()
    matplotlib.image.imsave(os.path.join(logdir, f"{filename}.png"), img)

    plt.figure(figsize=(8, 8))
    plt.title(filename.split('/')[-1])
    plt.imshow(img);
    plt.axis('off')
    plt.show()

def visual_eval(x, model, likelihood, logdir):
    build_and_save_grid(x, logdir, "original")

    # display posterior samples x ~ p(x|z), z ~ q(z|x)
    x_ = model(x).get('x_')

    if likelihood is not None:
        x_ = likelihood(logits=x_).sample()
    
    build_and_save_grid(x_, logdir, "posterior")

    # dislay prior samples x ~ p(x|z), z ~ p(z)
    x_ = model.sample_from_prior(100).get('x_')
    if likelihood is not None:
        x_ = likelihood(logits=x_).sample()
        
    build_and_save_grid(x_, logdir, "prior")

    print(logging_sep("="))
    print(f"Samples logged in {logdir}")
    
        # log to wandb
    for i in ['original', 'posterior', 'prior']:
        images = wandb.Image(
            f"{logdir}/{i}.png", 
            caption=f"{i} samples"
            )

        wandb.log({f"val_{i}": images})

def save_img(img, path):
    def _scale(img):
        img *= 255
        return img.astype(np.uint8)

    img = _scale(img)

    matplotlib.image.imsave(path, img)


def summary2logger(logger, summary, global_step, epoch, best=None, stats_key='loss'):
    """write summary to logging"""
    if not stats_key in summary.keys():
        logger.warning('key ' + str(stats_key) + ' not int output dictionary')
    else:
        message = f'\t[{global_step} / {epoch}]   '
        message += ''.join([f'{k} {v.item():6.2f}   ' for k, v in summary.get(stats_key).items()])
        message += f'({summary["info"]["elapsed-time"]:.2f}s /iter)'
        if best is not None:
            message += f'   (best: {best[0].item():6.2f}  [{best[1]} / {best[2]}])'
        logger.info(message)


def save_model(model, eval_summary, global_step, epoch, best_elbo, logdir, key='elbo'):
    elbo = eval_summary['loss'][key]
    prev_elbo, *_ = best_elbo
    if elbo > prev_elbo:
        best_elbo = (elbo, global_step, epoch)
        pth = os.path.join(logdir, "model.pth")
        torch.save(model.state_dict(), pth)

    return best_elbo


def load_model(model, logdir):
    device = next(iter(model.parameters())).device
    model.load_state_dict(torch.load(os.path.join(logdir, "model.pth"), map_location=device))


@torch.no_grad()
def sample_model(model, likelihood, logdir, global_step=0, writer=None, N=100):
    # sample model
    x_ = model.sample_from_prior(N).get('x_')

    if likelihood is not None:
        x_ = likelihood(logits=x_).sample()

    # make grid
    nrow = math.floor(math.sqrt(N))
    grid = make_grid(x_, nrow=nrow)

    # normalize
    grid -= grid.min()
    grid /= grid.max()

    # save the raw image
    img = grid.data.permute(1, 2, 0).cpu().numpy()
    matplotlib.image.imsave(os.path.join(logdir, "samples.png"), img)

    # log to wandb
    if writer is not None:
        images = wandb.Image(
            f"{logdir}/samples.png", 
            caption=f"val recons of best model at global step -{global_step}"
            )
    
        wandb.log({"val_recons": images})
        writer.add_image('samples', grid, global_step)
