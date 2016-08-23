# T-CNN: Tubelets with Convolution Neural Networks

## Introduction

The `TCNN` framework is a deep learning framework for object detection in videos. This framework was orginally designed for the [ImageNet VID](http://image-net.org/challenges/LSVRC/2015/index#vid) chellenge in ILSVRC2015.


## Citing T-CNN
If you are using the `T-CNN` code in you project, please cite the following works.

```latex
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
T-CNN is released under the MIT License.

## ImageNet 2015 VID detection results

| Track         | Validation Set   | Test Set   | Rank in ILSVRC2015   |
| :------------ | :--------------: | :--------: | :------------------: |
| Provided      | 73.8             | 67.8       | #1                   |
| Additional    | 77.0             | 69.7       | #2                   |

## Installations
### Prerequisites
1. [caffe](http://caffe.berkeleyvision.org) with `Python layer` and `pycaffe`
2. [GNU Parallel](http://www.gnu.org/software/parallel/)
3. [matutils](https://github.com/myfavouritekk/matutils)
4. [FCN tracker](https://github.com/scott89/FCNT)
5. `Matlab` with python [engine](http://www.mathworks.com/help/matlab/matlab-engine-for-python.html?refresh=true)

### Instructions
1. Clone the repository and sub-repositories from GitHub, let `$TCNN_ROOT` represents the root directory of the repository.

    ```bash
        $ # clone the repository
        $ git clone --recursive https://github.com/myfavouritekk/T-CNN.git
        $ cd $TCNN_ROOT
        $ # checkout the ImageNet 2015 VID branch
        $ git checkout ilsvrc2015vid
    ```

2. Compilation for `vdetlib`

    ```bash
        $ cd $TCNN_ROOT/vdetlib
        $ make
        $ export PYTHONPATH=$TCNN_ROOT/vdetlib:$PYTHONPATH
    ```
3. Download and install `caffe` in the `External` directory

    ```bash
        $ git clone https://github.com/BVLC/caffe.git External/caffe
        $ # modify `Makefile.config` and build with Python layer and pycaffe
        $ # detailed instruction, please follow http://caffe.berkeleyvision.org/installation.html
        $ export PYTHONPATH=$TCNN_ROOT/External/caffe/python:$PYTHONPATH
    ```

4. Download a modified version of [`FCN Tracker`](https://github.com/myfavouritekk/FCNT/tree/T-CNN) originally developed by Lijun Wang et. al.

    ```bash
        $ git clone --recursive -b T-CNN https://github.com/myfavouritekk/FCNT External/fcn_tracker_matlab
        $ # compile the caffe-fcn_tracking and configure FCNT
    ```

## Demo
1. Extract the sample data and still-image detection results

    ```bash
        $ cd $TCNN_ROOT
        $ unzip sample_data.zip -d data/
    ```

2. Generate optical flow for the videos

    ```bash
        $ mkdir ./data/opt_flow
        $ ls ./data/frames |
            parallel python tools/data_proc/gen_optical_flow.py ./data/frames/{} ./data/opt_flow/{} --merge
    ```

3. Multi-context suppression and motion-guided propagation in **Matlab**

    ```matlab
        >> addpath(genpath('tools/mcs_mgp'));
        >> mcs_mgp('data/opt_flow', 'data/scores', 'data/mcs_mgp')
    ```

4. Tubelet tracking and re-scoring

    ```bash
        $ # generate .vid protocol files
        $ ls data/frames | parallel python vdetlib/tools/gen_vid_proto_file.py {} $PWD/data/frames/{} data/vids/{}.vid
        $ # tracking from raw detection files
        $ find data/vids -type f -name *.vid | parallel -j1 python tools/tracking/greedy_tracking_from_raw_dets.py {} data/mcs_mgp/window_size_7_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow/{/.} data/tracks/{/.} --thres 3.15 --max_frames 100 --num 30
        $ # spatial max-pooling
        $ find data/vids -type f | parallel python tools/scoring/tubelet_raw_dets_max_pooling.py {} data/tracks/{/.} data/mcs_mgp/window_size_7_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow/{/.} data/score_proto/window_size_7_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow_max_pooling/{/.} --overlap_thres 0.5
    ```

5. Tubelet visualization

    ```bash
        $ python tools/visual/show_score_proto.py data/vids/ILSVRC2015_val_00007011.vid data/score_proto/window_size_7_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow_max_pooling/ILSVRC2015_val_00007011/ILSVRC2015_val_00007011.airplane.score
    ```

## Beyond demo
1. Optical flow extraction

    ```bash
        $ python tools data_proc/gen_optical_flow.py -h
    ```

2. [vdetlib](https://github.com/myfavouritekk/vdetlib) for tracking and rescoring
3. Visualization tools in `tools/visual`.


## Known Issues
1. Matlab engines may stall after long periods of tracking. Please consider to kill the certain matlab session to continue.

## To-do list
- [ ] Tubelet Bayesian classifier

