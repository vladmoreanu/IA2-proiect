import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, patch_size=128):
        self.noisy_filenames = sorted(os.listdir(noisy_dir))
        self.clean_filenames = sorted(os.listdir(clean_dir))

        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.noisy_filenames)

    def __getitem__(self, index):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_filenames[index])
        clean_path = os.path.join(self.clean_dir, self.clean_filenames[index])

        noisy_img = Image.open(noisy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')

        return self.transform(noisy_img), self.transform(clean_img)