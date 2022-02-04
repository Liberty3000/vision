
def build(obj, args, module, verbose=False):
    import inspect

    if not obj: return None
    if obj in args.keys(): args = args[obj]
    Object = getattr(module, obj)
    params = inspect.signature(Object).parameters.keys()
    params = {k:args[k] for k in params if k in args.keys()}
    if verbose: print('{}\n{}'.format(obj, params))
    return Object(**params)


def checkpoint(metrics, reports, splits=['train', 'vals']):
    for split in splits:
        print('-' * 80)
        for metric in metrics[split]:
            metric = '{} {}'.format(split, metric.__class__.__name__)
            if len(reports) != 1:
                sign = '+' if reports[-2][metric] < reports[-1][metric] else '-'
                diff = '{}{:.5f}'.format(sign, abs(reports[-1][metric] - reports[-2][metric]))
                banner = '{:15} | {:.5f} :: {} :: last: {:.5f}'
                print(banner.format(metric, reports[-1][metric], diff, reports[-2][metric]))
            else:
                print('{:15} | {:.5f}'.format(metric, reports[-1][metric]))


def log(split, metrics):
    import mlflow as mf, torchmetrics as tm

    report = dict()
    for metric in metrics[split]:
        value = metric.compute().detach().squeeze().item()
        key = '{} {}'.format(split, metric.__class__.__name__)
        report[key] = value
        if not isinstance(metric, tm.ConfusionMatrix): mf.log_metric(key, value)
        else:
            mf.log_metric('TP', value[0][0])
            mf.log_metric('TN', value[1][1])
            mf.log_metric('FP', value[1][0])
            mf.log_metric('FN', value[0][1])
    return report
