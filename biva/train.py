from tqdm import tqdm   
from biva.utils import Aggregator, summary2logger, save_model, sample_model
from biva.utils import LowerBoundedExponentialLR, logging_sep, available_device , training_step, test_step, summary2logger, save_model, load_model, sample_model, DiscretizedMixtureLogits
from booster.utils import EMA, logging_sep, available_device

def training_step(x, model, evaluator, optimizer, scheduler, **kwargs):
    optimizer.zero_grad()
    diagnostics = model(x, **kwargs)
    loss = -evaluator(diagnostics)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return diagnostics

def val_step(x, model, evaluator, **kwargs):
    model.eval()
    diagnostics = model(x, **kwargs)
    loss = -evaluator(diagnostics)
    model.train()
    return diagnostics

def train(model, ema, train_loader, val_loader, optimizer, scheduler, evaluator, train_aggregator, val_aggregator, EPOCHS, **kwargs):
    for epoch in range(1, EPOCHS + 1):

        # training
        train_aggregator.initialize()
        for x in tqdm(train_loader, desc='train epoch'):
            x = x.to('cuda')
            diagnostics = training_step(x, model, evaluator, optimizer, scheduler, **kwargs)
            train_aggregator.update(diagnostics)
            ema.update()
            global_step += 1
        train_summary = train_aggregator.data.to('cpu')

    # evaluation
    val_aggregator.initialize()
    for x in tqdm(val_loader, desc='val epoch'):
        x = x.to('cuda')
        diagnostics = val_step(x, ema.model, evaluator, **kwargs)
        val_aggregator.update(diagnostics)

    eval_summary = val_aggregator.data.to('cpu')

    # keep best model
    best_elbo = save_model(ema.model, eval_summary, global_step, epoch, best_elbo, logdir)

    # logging
    summary2logger(train_logger, train_summary, global_step, epoch)
    summary2logger(eval_logger, eval_summary, global_step, epoch, best_elbo)

    # wandb logging
    train_summary.log(train_writer, global_step)
    eval_summary.log(valid_writer, global_step)

    # sample model
    sample_model(ema.model, likelihood, logdir, writer=valid_writer, global_step=global_step, N=100)