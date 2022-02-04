import os
from pathlib import Path
import numpy as np, torch as th
from torch.utils.data import DataLoader
from vision.data import TensorTransformDataset
from sklearn.model_selection import KFold


metadata   = dict(
BloodMNIST = dict(image_shape=(3,28,28), num_classes=8),
PathMNIST  = dict(image_shape=(3,28,28), num_classes=9),
TissueMNIST= dict(image_shape=(1,28,28), num_classes=8)
)

def preprocess(subset, images, labels, normalization=False, zero_center=False):
    X = th.from_numpy(images).float()
    y = th.from_numpy(labels).long()
    X = X.view(-1, *metadata[subset]['image_shape'])
    y = y.view(-1)
    if normalization: X = (X - X.min()) / (X.max() - X.min())
    if zero_center: X = X - X.mean() / X.std() + 1e-9
    return X,y

def load(subset, splits, transforms=[], shuffle=True, normalization=True, zero_center=True, iterator=True, crossval_k=None, *args ,**kwargs):
    assert 'MEDMNIST_PATH' in os.environ.keys()
    path = os.path.join(os.environ['MEDMNIST_PATH'], subset.lower())
    bundle = dict(**metadata[subset])

    if crossval_k:
        raise NotImplementedError()
    else:
        for split,bsize in splits.items():
            if not bsize: continue

            imgs = np.load(Path(path) / '{}_images.npy'.format(split)).astype(np.float32)
            labs = np.load(Path(path) / '{}_labels.npy'.format(split)).astype(np.int64)

            X,y = preprocess(subset, imgs, labs, normalization, zero_center)

            data = TensorTransformDataset((X, y), transform=transforms[split])
            if iterator: data = DataLoader(data, batch_size=bsize, shuffle=shuffle)
            bundle[split] = data

    return bundle

def BloodMNIST(*args, **kwargs):
    return load(subset='BloodMNIST', *args ,**kwargs)
def PathMNIST(*args, **kwargs):
    return load(subset='PathMNIST', *args ,**kwargs)
def TissueMNIST(*args, **kwargs):
    return load(subset='TissueMNIST', *args ,**kwargs)
