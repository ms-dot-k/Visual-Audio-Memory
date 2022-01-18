# Multi-modality Associative Bridging through Memory. <br>Application in Lip Reading
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modality-associative-bridging-through/lipreading-on-lrw-1000)](https://paperswithcode.com/sota/lipreading-on-lrw-1000?p=multi-modality-associative-bridging-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modality-associative-bridging-through/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=multi-modality-associative-bridging-through)

**Please Do Not Fork This Repository**

This repository contains the official PyTorch implementation of the following paper:
> **Multi-modality Associative Bridging through Memory: Speech Sound Recollected from Face**<br>
> Minsu Kim*, Joanna Hong*, Sejin Park, and Yong Man Ro (\*Equal contribution)<br>
> Paper: https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Multi-Modality_Associative_Bridging_Through_Memory_Speech_Sound_Recollected_From_Face_ICCV_2021_paper.pdf<br>

<div align="center"><img width="90%" src="img/Img.png?raw=true" /></div>

## Preparation

### Requirements
- python 3.7
- pytorch 1.6 ~ 1.9
- torchvision
- torchaudio
- av
- tensorboard
- pillow

### Datasets
LRW dataset can be downloaded from the below link.
- https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

The pre-processing will be done in the data loader.<br>
The video is cropped with the bounding box \[x1:59, y1:95, x2:195, y2:231\].

## Training the Model
`main.py` saves the weights in `--checkpoint_dir` and shows the training logs in `./runs`.

To train the model, run following command:
```shell
# Distributed training example for LRW
python -m torch.distributed.launch --nproc_per_node='number of gpus' main.py \
--lrw 'enter_data_path' \
--checkpoint_dir 'enter_the_path_for_save' \
--batch_size 80 --epochs 200 \
--mode train --radius 16 --n_slot 88 \
--augmentations True --distributed True --dataparallel False\
--gpu 0,1...
```

```shell
# Data Parallel training example for LRW
python main.py \
--lrw 'enter_data_path' \
--checkpoint_dir 'enter_the_path_for_save' \
--batch_size 320 --epochs 200 \
--mode train --radius 16 --n_slot 88 \
--augmentations True --distributed False --dataparallel True \
--gpu 0,1...
```

Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint_dir`: directory for saving checkpoints
- `--batch_size`: batch size  `--epochs`: number of epochs  `--mode`: train / val / test
- `--augmentations`: whether performing augmentation  `--distributed`: Use DataDistributedParallel  `--dataparallel`: Use DataParallel
- `--gpu`: gpu for using `--lr`: learning rate `--n_slot`: memory slot size `--radius`: scaling factor for addressing score
- Refer to `main.py` for the other training parameters

## Testing the Model
To test the model, run following command:
```shell
# Testing example for LRW
python main.py \
--lrw 'enter_data_path' \
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 80 \
--mode test --radius 16 --n_slot 88 \
--test_aug True --distributed False --dataparallel False \
--gpu 0
```
Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint`: the checkpoint file
- `--batch_size`: batch size  `--mode`: train / val / test
- `--test_aug`: whether performing test time augmentation  `--distributed`: Use DataDistributedParallel  `--dataparallel`: Use DataParallel
- `--gpu`: gpu for using `--lr`: learning rate `--n_slot`: memory slot size `--radius`: scaling factor for addressing score
- Refer to `main.py` for the other testing parameters

## Pretrained Models
You can download the pretrained models. <br>
Put the ckpt in './data/'

**Bi-GRU Backend**
- https://drive.google.com/file/d/1wkgkRWxu7JM0uaNHmcyCpvVz9OFar8Do/view?usp=sharing <br>

To test the pretrained model, run following command:
```shell
# Testing example for LRW
python main.py \
--lrw 'enter_data_path' \
--checkpoint ./data/GRU_Back_Ckpt.ckpt \
--batch_size 80 --backend GRU\
--mode test --radius 16 --n_slot 88 \
--test_aug True --distributed False --dataparallel False \
--gpu 0
```

**MS-TCN Backend**
- https://drive.google.com/file/d/1uHZbmk9fgMqYVfvaoMUe-9XlGvQnXEcS/view?usp=sharing

To test the pretrained model, run following command:
```shell
# Testing example for LRW
python main.py \
--lrw 'enter_data_path' \
--checkpoint ./data/MSTCN_Back_Ckpt.ckpt \
--batch_size 80 --backend MSTCN\
--mode test --radius 16 --n_slot 168 \
--test_aug True --distributed False --dataparallel False \
--gpu 0
```

|       Architecture      |   Acc.   |
|:-----------------------:|:--------:|
|Resnet18 + MS-TCN + Multi-modal Mem   |   85.864    |
|Resnet18 + Bi-GRU + Multi-modal Mem   |   85.408    |

## AVSR
You can also use the pre-trained model to perform Audio Visual Speech Recognition (AVSR), since it is trained with both audio and video inputs. <br>
In order to use AVSR, just use ''tr_fusion'' (refer to the train code) for prediction.

## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{kim2021multimodalmem,
  title={Multi-Modality Associative Bridging Through Memory: Speech Sound Recollected From Face Video},
  author={Kim, Minsu and Hong, Joanna and Park, Se Jin and Ro, Yong Man},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={296--306},
  year={2021}
}
```
