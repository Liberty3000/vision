import click, functools, importlib, inspect, os, pathlib, pprint, tqdm, types, yaml
import mlflow as mf, numpy as np, pandas as pd
from ray import tune
import torch as th, torchmetrics as tm, torchvision as tv


def build(obj, module, args):
    Object = getattr(module, obj)
    params = inspect.signature(Object).parameters.keys()
    params = {k:args[k] for k in params if k in args.keys()}
    return Object(**params)

def train(checkpoint_dir=None, *args, **kwargs):
    conf = types.SimpleNamespace(**kwargs)

    base_transform = [tv.transforms.ToTensor()]
    transforms = dict(train=tv.transforms.Compose(base_transform + [
    tv.transforms.RandomVerticalFlip(conf.random_vflip),
    tv.transforms.RandomHorizontalFlip(conf.random_hflip)]),
    val=tv.transforms.Compose(base_transform),
    test=tv.transforms.Compose(base_transform))

    bundle = vision.data.datasets[conf.dataset](splits=dict(train=conf.batch_size,
    val=conf.batch_size), transforms=transforms, split_ratio=conf.trainval_split,
    normalize=conf.normalize, standardize=conf.standardize)
    trainer, validator = bundle['train'], bundle['val']

    with open(conf.config, 'r') as f: config = yaml.safe_load(f)
    params = {**kwargs, **bundle, **config}
    pprint.pprint(params, indent=2)

    with mf.start_run() as run:
        run_id = mf.active_run().info.run_id
        print('{}\n{} :: {}'.format('-' * 80, conf.dataset, run_id))
        run_dir = os.path.join(conf.outdir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'conf.yaml'), 'w') as f:
            yaml.dump({**kwargs, **config}, f)
        ckpt_save = os.path.join(run_dir, 'ckpt.{}.sd')
        step_save = os.path.join(run_dir, 'epoch.{step}.sd')
        report_file = os.path.join(run_dir, 'metrics.csv')

        ########################################################################
        module, class_ = '.'.join(conf.model.split('.')[:-1]), conf.model.split('.')[-1]
        model = build(class_, importlib.import_module(module), params).to(conf.device)

        criterion = build(params['loss'], th.nn, params)
        loss_fn = criterion.__class__.__name__

        opt = build(params['optim'], th.optim, dict(params=model.parameters(), **params))

        metrics = dict(train=[build(metric, tm, params) for metric in conf.metrics],
                         val=[build(metric, tm, params) for metric in conf.metrics])
        ########################################################################
        if checkpoint_dir is not None and os.path.isdir(checkpoint_dir):
            model_sd, opt_sd = th.load(os.path.join(checkpoint_dir, 'checkpoint'))
            model.load_state_dict(model_sd)
            opt.load_state_dict(opt_sd)
        ########################################################################
        reports = []
        for epoch in range(1,1 + conf.epochs):
            report = dict()
            pbar = tqdm.tqdm(enumerate(trainer), total=len(trainer))
            for itr,(images,labels) in pbar:
                images,labels = images.to(conf.device), labels.to(conf.device)

                output = model(images)
                labels = labels.long().squeeze()
                loss = criterion(output, labels)

                opt.zero_grad()
                loss.backward()
                opt.zero_grad()

                mf.log_metric('train {}'.format(loss_fn), loss.item())
                pbar.set_description('{}: {:.5f}'.format(loss_fn, loss.item()))

                for metric in metrics['train']:
                    metric(output.cpu().detach(), labels.cpu().detach())
                    metric_ = 'train {}'.format(metric.__class__.__name__)
                    value = metric.compute().detach().squeeze().item()
                    report[metric_] = value
                    mf.log_metric(metric_, value)

                if conf.test and itr == 1e1: break
            ####################################################################
            if 'val' in bundle.keys():
                pbar = tqdm.tqdm(enumerate(validator), total=len(validator))
                total_loss = 0
                for itr,(images,labels) in pbar:
                    images,labels = images.to(conf.device), labels.to(conf.device)

                    output = model(images)
                    labels = labels.long().squeeze()
                    loss = criterion(output, labels)
                    total_loss += loss.item()

                    if conf.test and itr == 1e0: break

                loss = total_loss / len(validator)
                mf.log_metric('val {}'.format(loss_fn), loss)
                pbar.set_description('{}: {:.5f}'.format(loss_fn, loss))
                for metric in metrics['val']:
                    metric(output.cpu().detach(), labels.cpu().detach())
                    metric_ = 'val {}'.format(metric.__class__.__name__)
                    value = metric.compute().detach().squeeze().item()
                    report[metric_] = value
                    mf.log_metric(metric_, value)

                print('Epoch {}'.format(epoch))
                pprint.pprint(report)

                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    save_as = step_save.format(step=str(epoch).zfill(6))
                    th.save((model.state_dict(), opt.state_dict()), save_as)
                reports.append(report)
    pd.DataFrame(reports).to_csv(report_file, index=False)
    tune.report(**report)

@click.command()
@click.option(             '--n', default=1)
@click.option(          '--seed', default=0)
@click.option(        '--device', default='cuda:0')
@click.option(          '--test', default=False, is_flag=True)
################################################################################
@click.option(         '--model', required=True)
@click.option(        '--config', required=True)
################################################################################
@click.option(       '--dataset', required=True)
@click.option(     '--normalize', default=True, type=bool)
@click.option(   '--standardize', default=True, type=bool)
################################################################################
@click.option(       '--metrics', default=['Accuracy','Precision','Recall','F1','AUROC'], type=list)
@click.option(        '--target', default='val AUROC')
@click.option(          '--mode', default='max')
@click.option(          '--loss', default='CrossEntropyLoss')
################################################################################
@click.option(        '--epochs', default=10)
@click.option(    '--batch_size', default=2**9)
@click.option(      '--interval', default=1000)
@click.option('--trainval_split', default=None)
################################################################################
@click.option(  '--random_hflip', default=0.5, type=float)
@click.option(  '--random_vflip', default=0.5, type=float)
################################################################################
@click.option(        '--outdir', default=os.getcwd())
@click.pass_context
def run(ctx, **args):
    mf.set_experiment(args['dataset'])
    result = tune.run(functools.partial(train, **args), resources_per_trial=dict(cpu=8,
    gpu=1), metric=args['target'], mode=args['mode'], num_samples=args['n'])
    ############################################################################
    best = result.get_best_trial(args['target'])
    pprint.pprint(best.config)
    print('{}\n{} :: {} :: {}'.format('-' * 80 ,run.info.run_id,
    args['target'], best.last_result[args['target']]))

if __name__ == '__main__':
    run()
