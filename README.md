![torch](https://img.shields.io/static/v1?label=torch&message=1.10.0&color=3258a8)
![torchmetrics](https://img.shields.io/static/v1?label=torchmetrics&message=0.6.0&color=73568f)
![torchvision](https://img.shields.io/static/v1?label=torchvision&message=0.11.1&color=32a898)
![einops](https://img.shields.io/static/v1?label=einops&message=0.3.2&color=45a0e6)
## Vision
Building blocks for computer vision research.
___
### Models

#### Vision Transformer [![arxiv](https://img.shields.io/badge/arXiv-2010.11929-maroon.svg)](https://arxiv.org/abs/2010.11929)

<p align="center">
  <img src="docs/ViT.png" />
</p>


```bash
python -m vision.train --model='vit.ViT' ...
```

#### ResMLP [![arxiv](https://img.shields.io/badge/arXiv-2105.03404-maroon.svg)](https://arxiv.org/abs/2105.03404)

<p align="center">
  <img src="docs/ResMLP.png" />
</p>

```bash
python -m vision.train --model='resmlp.ResMLP' ...
```

#### MLPMixer [![arxiv](https://img.shields.io/badge/arXiv-2105.01601-maroon.svg)](https://arxiv.org/abs/2105.01601)

<p align="center">
  <img src="docs/MLPMixer.png" width="768" height="348"/>
</p>

```bash
python -m vision.train --model='mlpmixer.MLPMixer' ...
```

#### ConvMixer [![arxiv](https://img.shields.io/badge/arXiv-2201.09792-maroon.svg)](https://arxiv.org/abs/2201.09792)
WIP

#### ConvNeXt [![arxiv](https://img.shields.io/badge/arXiv-2201.03545-maroon.svg)](https://arxiv.org/abs/2201.03545)
WIP

___

## Datasets

- #### MedMNISTv2
- #### PascalVOC
- #### Tiny ImageNet

___
## Experiments

WIP
