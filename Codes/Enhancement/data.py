import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, CenterCrop
from PIL import Image
import random


class TerahertzDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.lr_image_files = sorted(os.listdir(os.path.join(root_dir, 'lr_resized_images')))
        self.hr_image_files = sorted(os.listdir(os.path.join(root_dir, 'hr_images')))

        self.lr_image_files = [os.path.join(self.root_dir, 'lr_resized_images', x) for x in self.lr_image_files]
        self.hr_image_files = [os.path.join(self.root_dir, 'hr_images', x) for x in self.hr_image_files]

        self.data_len = len(self.lr_image_files)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        lr_image = Image.open(self.lr_image_files[idx])
        hr_image = Image.open(self.hr_image_files[idx])

        # Crop the images to 448 x 1216
        lr_image = CenterCrop((1216, 448))(lr_image)
        hr_image = CenterCrop((1216, 448))(hr_image)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


def train_val_test_split(dataset, train_percent=.6, val_percent=.2, test_percent=.2, shuffle=True):
    assert train_percent + val_percent + test_percent == 1.0
    data_len = len(dataset)
    indices = list(range(data_len))
    if shuffle:
        random.shuffle(indices)
    train_end = int(train_percent * data_len)
    val_end = int(val_percent * data_len) + train_end
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


data_transform = Compose([ToTensor()])
dataset = TerahertzDataset(root_dir='D:\THz\enhancement_pytorch', transform=data_transform)
train_indices, val_indices, test_indices = train_val_test_split(dataset, train_percent=.6, val_percent=.2,
                                                                test_percent=.2, shuffle=True)
