import tqdm
import torch as th, torchvision as tv

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

def iterator(generator, device='cuda:0'):
    pbar = tqdm.tqdm(enumerate(generator), total=len(generator))
    for itr,(images,labels) in pbar:
        pbar.update(1)
        images,labels = images.to(device), labels.to(device).long()
        yield images, labels

from vision.data.MedMNISTv2 import *
datasets = dict(PathMNIST=PathMNIST, TissueMNIST=TissueMNIST, BloodMNIST=BloodMNIST)
