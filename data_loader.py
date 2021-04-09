import os
import random
import pickle
import torch

from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from natsort import natsorted


class CACDLoader(data.Dataset):
    def __init__(self, image_dir, num_classes, mode, transform=None):
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.mode = mode
        self.transform = transform

        self.train_dataset = pickle.load(open(self.image_dir + 'splits.pkl', 'rb'))['train']
        self.transfer_dataset = natsorted(list(os.listdir(os.path.join(self.image_dir, 'transfer'))))

        random.seed(41)  # dirk's jersey
        random.shuffle(self.train_dataset)

        if mode == 'train':
            self.ims = self.train_dataset
        elif mode == 'val':
            self.ims = self.val_dataset
        elif mode == 'transfer':
            self.ims = self.transfer_dataset
        elif mode == 'test':
            self.ims = self.test_dataset
        else:
            print('dataset type not found: {}'.format(mode))
            raise NotImplementedError

    def get_age_class(self, age):
        lab = -1
        if age <= 30:
            lab = 0
        elif age > 30 and age <= 40:
            lab = 1
        elif age > 40 and age <= 50:
            lab = 2
        elif age > 50:
            lab = 3

        if lab == -1:
            print('found unknown label: ', age)
            exit(0)
        else:
            return lab

    def __getitem__(self, index):
        item = self.ims[index]
        if self.ims is not self.transfer_dataset:
            label = int(item[1])
            path = '{}{}{}'.format(self.image_dir, 'train/', item[0].split('/')[-1])

            # hard-code for age classes
            label = self.get_age_class(label)

            # one-hot
            label_oh = torch.eye(self.num_classes)[label]

            image = Image.open(path).convert('RGB')

            return self.transform(image), label_oh, label
        else:
            path = '{}{}/{}'.format(self.image_dir, 'transfer', item)
            image = Image.open(path).convert('RGB')

            return self.transform(image)

    def __len__(self):
        return len(self.ims)


def get_loader(image_dir, image_size=128, batch_size=32, num_classes='4', mode='train', num_workers=8):
    """Build and return a data loader."""
    transform = []

    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CACDLoader(image_dir, num_classes, mode, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader
