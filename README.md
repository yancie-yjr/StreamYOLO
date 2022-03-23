# StreamYOLO

## Real-time Object Detection for Streaming Perception
<p align='left'>
  <img src='example.gif' width='721'/>
</p>

[Chenxu Luo](https://chenxuluo.github.io/), [Xiaodong Yang](https://xiaodongyang.org/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/) <br>
Exploring Simple 3D Multi-Object Tracking for Autonomous Driving, ICCV 2021<br>
[[Paper]](https://arxiv.org/pdf/2108.10312.pdf) [[Poster]](poster.pdf) [[YouTube]](https://www.youtube.com/watch?v=awK1O-wf_74)

## Getting Started
### Installation
Please refer to [INSTALL](INSTALL.md) for the detail.

### Data Preparation 
* [nuScenes](https://www.nuscenes.org)
```
python ./tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```
* [Waymo Open Dataset](https://waymo.com/open/) (TODO)

### Training
```
python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/point_pillars/configs/nusc_all_pp_centernet_tracking.py --work_dir SAVE_DIR
```

### Test
In `./model_zoo` we provide our trained (pillar based) model on nuScenes.          
Note: We currently only support inference with a single GPU.
```
python ./tools/val_nusc_tracking.py examples/point_pillars/configs/nusc_all_pp_centernet_tracking.py --checkpoint CHECKPOINTFILE  --work_dir SAVE_DIR
```

## Citation
Please cite the following paper if this repo helps your research:
```bibtex
@InProceedings{Luo_2021_ICCV,
    author    = {Luo, Chenxu and Yang, Xiaodong and Yuille, Alan},
    title     = {Exploring Simple 3D Multi-Object Tracking for Autonomous Driving},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year      = {2021}
}
```

## License
Copyright (C) 2021 QCraft. All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (Attribution-NonCommercial-ShareAlike 4.0 International). The code is released for academic research use only. For commercial use, please contact [business@qcraft.ai](business@qcraft.ai).
