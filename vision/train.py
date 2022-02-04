import click, os, pprint, tqdm, types, yaml
from types import SimpleNamespace as Namespace
import mlflow as mf, numpy as np, pandas as pd
import torch as th, torchmetrics as tm, torchvision as tv
from torchvision.transforms import Compose

from neurpy.util import enforce_reproducibility
from vision.data import datasets
from vision.model import load
from vision.util import build, checkpoint, log


def train(trainer, validator=None, **args):
    conf = Namespace(**args)
    #---------------------------------------------------------------------------
    dir = os.path.join(conf.outdir, mf.active_run().info.run_id)
    run = Namespace(id=mf.active_run().info.run_id, dir=dir, **{k:os.path.join(dir,v)
    for k,v in dict(config='conf.yaml',report='metrics.csv', batch_ckpt='batch.{}.sd',
    epoch_ckpt='epoch.{}.sd').items()})

    with open(conf.config, 'r') as f: config = yaml.safe_load(f)
    params = {**args, **config}

    os.makedirs(run.dir, exist_ok=True)
    os.chdir(run.dir)

    with open(run.config, 'w') as f: yaml.dump(params, f)
    pprint.pprint(params, indent=2)
    print('{}\n{} {} :: {}'.format('-' * 80, conf.dataset, conf.model, run.id))
    #---------------------------------------------------------------------------
    # load torch.nn.Module by name
    model = load(params['model'], params).to(conf.device)

    # load loss from torch.nn by name
    criterion = build(params['loss'], params, th.nn)
    loss_fn = criterion.__class__.__name__

    # load optimizer from torch.optim
    opt = build(params['optim'], dict(params=model.parameters(), **params), th.optim)

    # load learning rate scheduler from torch.optim.lr_scheduler
    scheduler = build(params['scheduler'], dict(optimizer=opt, **params), th.optim.lr_scheduler)

    # load metrics from torchmetrics
    metrics = dict(train=[build(metric, params, tm) for metric in conf.metrics],
                     val=[build(metric, params, tm) for metric in conf.metrics])
    #---------------------------------------------------------------------------
    if conf.init:
        print('restoring weights from `{}`...')
        model_sd,_ = th.load(conf.init)
        model.load_state_dict(model_sd)
        if isinstance(_, type(model_sd)):
            opt.load_state_dict(_)
        print('[init] :: {}'.format(conf.init))
    #---------------------------------------------------------------------------
    reports = []
    for epoch in range(1,1 + conf.epochs):
        split, report, val_report = 'train', {}, {}
        model.train()
        pbar = tqdm.tqdm(enumerate(trainer), total=len(trainer))
        for itr,(images,labels) in pbar:

            # forward
            images,labels = images.to(conf.device), labels.to(conf.device).long()
            output = model(images)
            loss = criterion(output, labels)

            # backward
            opt.zero_grad()
            loss.backward()
            if conf.clip_grad_norm:
                th.nn.utils.clip_grad_norm_(model.parameters(), conf.clip_grad_norm)
            opt.step()

            # log training metrics
            pbar.set_description('{}: {:.5f}'.format(loss_fn, loss.item()))
            mf.log_metric('{} {}'.format(split, loss_fn), loss.item())
            for metric in metrics[split]:
                metric.update(output.cpu().detach(), labels.cpu().detach())
            report = log(split, metrics)

            # batch interval callback
            if itr and itr % conf.batch_interval == 0:
                save_as = run.batch_ckpt.format(str(itr * conf.epochs).zfill(6))
                th.save((model.state_dict(), opt.state_dict(), epoch), save_as)

            # short-circuit training
            if conf.test and itr == 1e1: break
    #---------------------------------------------------------------------------
        if validator is not None:
            split = 'val'
            model.eval()
            pbar, total_loss = tqdm.tqdm(enumerate(validator), total=len(validator)), 0
            for itr,(images,labels) in pbar:

                # forward
                images,labels = images.to(conf.device), labels.to(conf.device).long()
                output = model(images)
                loss = criterion(output, labels)
                total_loss += loss.item()

                # short-circuit validation
                if conf.test and itr == 1e1: break
            loss = total_loss / len(validator)

            # log validation metrics
            pbar.set_description('{}: {:.5f}'.format(loss_fn, loss))
            mf.log_metric('{} {}'.format(split, loss_fn), loss)
            for metric in metrics[split]:
                metric.update(output.cpu().detach(), labels.cpu().detach())
            val_report = log(split, metrics)
    #---------------------------------------------------------------------------
        print('Epoch {} :: {}'.format(epoch, run.id))
        reports.append({**report, **val_report})
        checkpoint(metrics, reports)

        if epoch % conf.epoch_interval == 0:
            save_as = run.epoch_ckpt.format(str(epoch).zfill(6))
            th.save((model.state_dict(), opt.state_dict(), epoch), save_as)

        df = pd.DataFrame(reports).to_csv(run.report, index=False)
    #---------------------------------------------------------------------------
    return model, df

@click.command()
@click.option(          '--seed', default=0)
@click.option(        '--device', default='cuda:0')
@click.option(          '--test', default=False, is_flag=True)
@click.option(        '--outdir', default=os.getcwd())
#-------------------------------------------------------------------------------
@click.option(        '--config')
@click.option(         '--model', required=True)
@click.option(       '--dataset', required=True)
#-------------------------------------------------------------------------------
@click.option( '--normalization', default=False, type=bool)
@click.option(   '--zero_center', default=True,  type=bool)
@click.option(    '--zca_whiten', default=False, type=bool)
#-------------------------------------------------------------------------------
@click.option(  '--random_hflip', default=0.5, type=float)
@click.option(  '--random_vflip', default=0.5, type=float)
#-------------------------------------------------------------------------------
@click.option(    '--crossval_k', default=None)
@click.option(       '--metrics', default=['Accuracy','Precision','Recall','F1','AUROC'], type=list)
@click.option(          '--loss', default='CrossEntropyLoss')
#-------------------------------------------------------------------------------
@click.option(         '--optim', default='Adam')
@click.option(            '--lr', default=5e-4)
@click.option(  '--weight_decay', default=1e-4)
@click.option('--clip_grad_norm', default=None)
@click.option(     '--scheduler', default=None)
#-------------------------------------------------------------------------------
@click.option(          '--init', default=None)
@click.option(        '--epochs', default=2**5)
@click.option('--epoch_interval', default=2**5)
@click.option(    '--batch_size', default=2**9)
@click.option('--batch_interval', default=2**5)
@click.option('--trainval_split', default=None)
@click.pass_context
def run(ctx, **args):
    enforce_reproducibility(args['seed'])
    mf.set_experiment('.'.join(args['dataset']))

    base, augs = [tv.transforms.ToTensor()], []
    augs += [tv.transforms.RandomVerticalFlip(args['random_vflip'])]
    augs += [tv.transforms.RandomHorizontalFlip(args['random_hflip'])]
    transforms = dict(train=Compose(base + augs), val=Compose(base))

    bundle = datasets[args['dataset']](
    splits=dict(train=args['batch_size'], val=args['batch_size']),
    transforms=transforms, split_ratio=args['trainval_split'],
    normalization=args['normalization'], zca_whiten=args['zca_whiten'],
    zero_center=args['zero_center'], crossval_k=args['crossval_k'])
    #---------------------------------------------------------------------------
    if not args['crossval_k']:
        trainer, validator = bundle['train'], bundle['val']
        del bundle['train']
        del bundle['val']
        with mf.start_run() as run:
            train(trainer=trainer, validator=validator, **args, **bundle)
    else:
        for fold in range(args['crossval_k']):
            print('{}\nFold {}'.format('-' * 80, fold))
            trainer = bundle['k{}_train'.format(fold)]
            validator = bundle['k{}_val'.format(fold)]
            with mf.start_run() as run:
                folds.append(run.info.run_id)
                train(trainer=trainer, validator=validator, **args, **bundle)
    #---------------------------------------------------------------------------

if __name__ == '__main__':
    run()
