# T-CNN: Tubelets with Convolution Neural Networks

## Introduction

## Citing T-CNN
If you are using the `T-CNN` code in you project, please cite the following works.

```bib
    @inproceedings{kang2016object,
      Title = {Object Detection from Video Tubelets with Convolutional Neural Networks},
      Author = {Kang, Kai and Ouyang, Wanli and Li, Hongsheng and Wang, Xiaogang},
      Booktitle = {CVPR},
      Year = {2016}
    }
    @article{kang2016tcnn,
      title={T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos},
      author={Kang, Kai and Li, Hongsheng and Yan, Junjie and Zeng, Xingyu and Yang, Bin and Xiao, Tong and Zhang, Cong and Wang, Zhe and Wang, Ruohui and Wang, Xiaogang and Ouyang, Wanli},
      journal={arXiv preprint},
      year={2016}
    }
```

## License

## ImageNet VID detection results

## Installations
### Prerequisites
- [caffe](http://caffe.berkeleyvision.org)

### Instruction
1. clone the repository and sub-repositories from GitHub, let `$TCNN_ROOT` represents the root directory of the repository.
```shell
    $ # clone the repository
    $ git clone --recursive https://github.com/myfavouritekk/T-CNN.git
    $ cd $TCNN_ROOT
    $ # checkout the ImageNet 2015 VID branch
    $ git checkout ilsvrc2015vid
```
2. compilation for `vdetlib`
```shell
    $ cd $TCNN_ROOT/vdetlib
    $ make
```

## Demo


