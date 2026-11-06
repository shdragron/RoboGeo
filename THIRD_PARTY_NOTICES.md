# Third-party notices

The RoboGeo code is released under the [Apache License 2.0](LICENSE). This repository also
contains third-party code, listed below. Third-party code keeps its own license. Each vendored
folder holds the upstream license file, and every upstream file we changed starts with a
"Modified by the RoboGeo authors" line.

## Vendored code

| component | location | license |
|---|---|---|
| [BEVFormer](https://github.com/fundamentalvision/BEVFormer) @ 66b65f3 | `detection/bevformer` | Apache-2.0 ([LICENSE](detection/bevformer/LICENSE)) |
| [CAPE](https://github.com/kaixinbear/CAPE) @ d7f0e0ba, Copyright (c) 2022 Megvii Inc. | `detection/cape` | Apache-2.0 ([LICENSE](detection/cape/LICENSE)) |
| [PETR](https://github.com/megvii-research/PETR), on which CAPE is built | `detection/cape` | Apache-2.0 |
| [DETR3D](https://github.com/WangYueFt/detr3d), Copyright (c) 2021 Wang, Yue: parts of the plugin code of BEVFormer and CAPE | `detection/bevformer/projects/mmdet3d_plugin/core/bbox`, `detection/cape/projects/mmdet3d_plugin` | MIT ([text below](#detr3d-mit-license)) |
| [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [mmdetection](https://github.com/open-mmlab/mmdetection), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [mmcv](https://github.com/open-mmlab/mmcv), Copyright (c) OpenMMLab: adapted files and the patches in `env/*/patches` | `detection/`, `env/` | Apache-2.0 |
| [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit): patches in `env/*/patches` | `env/` | Apache-2.0 |
| [GaussianLSS](https://github.com/HCIS-Lab/GaussianLSS) @ cd6d305, Copyright (c) 2025 HCIS Lab | `segmentation/third_party/GaussianLSS` | MIT ([LICENSE](segmentation/third_party/GaussianLSS/LICENSE)) |
| [Cross View Transformers](https://github.com/bradyz/cross_view_transformers) @ 4de6e641, Copyright (c) 2022 Brady Zhou | `segmentation/third_party/cross_view_transformers` | MIT ([LICENSE](segmentation/third_party/cross_view_transformers/LICENSE)) |
| [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization), Inria and MPII, as modified by GaussianLSS | `segmentation/third_party/GaussianLSS/GaussianLSS/model/diff-gaussian-rasterization` | Gaussian-Splatting License, **non-commercial research and evaluation only** ([LICENSE.md](segmentation/third_party/GaussianLSS/GaussianLSS/model/diff-gaussian-rasterization/LICENSE.md)) |
| [GLM](https://github.com/g-truc/glm), bundled with the rasterizer | `.../diff-gaussian-rasterization/third_party/glm` | Happy Bunny License or MIT ([copying.txt](segmentation/third_party/GaussianLSS/GaussianLSS/model/diff-gaussian-rasterization/third_party/glm/copying.txt)) |
| [stb_image_write](https://github.com/nothings/stb), Sean Barrett, bundled with the rasterizer | `.../diff-gaussian-rasterization/third_party/stbi_image_write.h` | public domain or MIT (in the file) |

GaussianLSS needs the Gaussian rasterizer for both training and inference. The GaussianLSS code
path and its checkpoints can therefore only be used for non-commercial research and evaluation.
CVT does not use the rasterizer.

## Downloaded at build or run time

These files are not included in this repository:

| file | used by | license |
|---|---|---|
| ResNet-50 ImageNet weights (`torchvision://resnet50`) | BEVFormer training | BSD-3-Clause ([torchvision](https://github.com/pytorch/vision)) |
| `resnet50_msra-5891d200.pth`, converted by OpenMMLab from the [MSRA ResNet release](https://github.com/KaimingHe/deep-residual-networks) | CAPE training | MIT |
| EfficientNet-B4 ImageNet weights, via [efficientnet_pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) | GaussianLSS, CVT | Apache-2.0 |
| `nvidia/cuda` base images of the Dockerfiles | all Docker images | [NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf) |

## DETR3D MIT license

```
MIT License

Copyright (c) 2021 Wang, Yue

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
