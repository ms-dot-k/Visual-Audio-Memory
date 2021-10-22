# Multi-modality Associative Bridging through Memory. <br>Application in Lip Reading : Cross-modal Memory Augmented Visual Speech Recognition
This repository contains the official PyTorch implementation of the following papers:
> **Multi-modality Associative Bridging through Memory: Speech Sound Recollected from Face**<br>
> Minsu Kim, Joanna Hong, Sejin Park, and Yong Man Ro<br>
> Paper: https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Multi-Modality_Associative_Bridging_Through_Memory_Speech_Sound_Recollected_From_Face_ICCV_2021_paper.pdf<br>

> **CroMM-VSR: Cross-Modal Memory Augmented Visual Speech Recognition**<br>
> Minsu Kim, Joanna Hong, Sejin Park, and Yong Man Ro<br>
> Paper: https://ieeexplore.ieee.org/abstract/document/9566778<br>

## Preparation

### Requirements
- python 3.7
- pytorch 1.6+
- opencv-python
- torchvision
- torchaudio
- av
- tensorboard

### Datasets
LRW dataset can be downloaded from the below link.
- https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

We don't have to preprocess the video into images.<br>
We process it in the data_loader.<br>
The video is cropped with the bounding box \[x1:59, y1:95, x2:195, y2:231\]

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
- Refer to `train.py` for the other training parameters

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
- Refer to `test.py` for the other testing parameters

## Pretrained Models
You can download the pretrained models.
- 

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

@article{kim2021cromm,
  title={CroMM-VSR: Cross-Modal Memory Augmented Visual Speech Recognition},
  author={Kim, Minsu and Hong, Joanna and Park, Se Jin and Ro, Yong Man},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
```
