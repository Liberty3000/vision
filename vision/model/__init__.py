
def load(model, params=dict(), pretrained=True, **kwargs):
    import importlib
    import torch as th, torchvision as tv

    # load torch.nn.Module by name
    if 'torchvision' in model:
        model = getattr(model, tv.models)(pretrained=pretrained)
    elif 'hub:' in model:
        # 'hub:hustvl/yolop.yolop' <- th.hub.load('hustvl/yolop', 'yolop', ...)
        model = th.hub.load(*conf.model[4:].split('.'), pretrained=pretrained)
    else:
        from vision.util import build
        imports = 'vision.model.{}'.format('.'.join(model.split('.')[:-1]))
        modules = importlib.import_module(imports)
        model = build(model.split('.')[-1], params, modules)

    return model
