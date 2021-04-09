# ''Enhancing Facial Data Diversity With Style-Based Face Aging''

[[paper](https://arxiv.org/abs/2006.03985)]

## run

```bash
python main.py --mode train --num_classes 4
```

## data folder structure

- `data-cacd/train` must contain the entire dataset (splits.pkl specifies the training and testing splits).
- `data-cacd/transfer` must contain 4 images ordered numerically--one for each age class. These are used as the target images from which we extract age styles during validation. We recommend placing the the following choice of target images into the `data-cacd/transfer` folder:

```bash
data-cacd/transfer
├── 15_Nicole_Gale_Anderson_0004.jpg
├── 35_Kate_Winslet_0007.jpg
├── 45_Tate_Donovan_0007.jpg
└── 61_Brian_George_0008.jpg
```


## citation

```bash
@InProceedings{Georgopoulos_2020_CVPR_Workshops,
author = {Georgopoulos, Markos and Oldfield, James and Nicolaou, Mihalis A. and Panagakis, Yannis and Pantic, Maja},
title = {Enhancing Facial Data Diversity With Style-Based Face Aging},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
```

## thanks

The codebase borrows heavily from [StarGAN](https://github.com/yunjey/stargan)--thank you!
