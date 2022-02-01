import os
import numpy as np, torch as th, torchvision as tv

class TensorTransformDataset(th.utils.data.Dataset):
    def __init__(self, tensors, transform):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors, self.transform = tensors, transform
        self.pil = tv.transforms.ToPILImage()

    def __getitem__(self, idx):
        x = self.tensors[0][idx]
        if self.transform is not None:
            x = self.transform(self.pil(x))
        return x, self.tensors[1][idx]

    def __len__(self):
        return self.tensors[0].size(0)

def PathMNIST(splits, transforms=[], shuffle=True, normalize=True, standardize=True, iterator=True, *args ,**kwargs):
    bundle = dict(image_shape=(3,28,28), num_classes=9)

    assert 'PATHMNIST_PATH' in os.environ.keys()
    path = os.environ['PATHMNIST_PATH']

    for split,bsize in splits.items():
        if not bsize: continue

        imgs = os.path.join(path, '{}_images.npy'.format(split))
        labs = os.path.join(path, '{}_labels.npy'.format(split))
        X = th.from_numpy(np.load(imgs).astype(np.float32)).float()
        y = th.from_numpy(np.load(labs).astype(np.int64)).long()

        X = X.view(-1, 3, 28, 28)
        y = y.view(-1, 1)

        if normalize: X = (X - X.min()) / (X.max() - X.min())
        if standardize: X = X - X.mean() / X.std() + 1e-9

        data = TensorTransformDataset((X, y), transform=transforms[split])
        if iterator: data = th.utils.data.DataLoader(data, batch_size=bsize, shuffle=shuffle)
        bundle[split] = data

    return bundle

datasets = dict(PathMNIST=PathMNIST)
