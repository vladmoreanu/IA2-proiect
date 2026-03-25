from model import DnCNN
from dataset import ImageDataset

import lighter
import torch
from torch.utils.data import DataLoader, random_split

# path_clean = r'./DATASETS/Flickr2K/normal_images'
# path_noisy = r'./DATASETS/Flickr2K/noise_images'

path_clean = r"E:\baze de date\Flickr2K\normal_images_tiles"
path_noisy = r"E:\baze de date\Flickr2K\noise_images_tiles"


if __name__ == '__main__':
    batch_size = 16
    train_split = 0.8
    num_of_layers = 17  # 9  # default 17
    epochs = 20
    learning_rt = 1e-3

    num_workers = 4
    prefetch_factor = 6

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'System :')
    print(f'  PyTorch version: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    print(f'  CUDA version PyTorch expects: {torch.version.cuda}')
    print(f'  Using device: {device}, {torch.device(device)}')

    dataset = ImageDataset(
        noisy_dir=path_noisy,
        clean_dir=path_clean
    )

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # same split across different runs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory_device=device,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory_device=device,
        pin_memory=True,
        persistent_workers=True
    )

    model = DnCNN(num_of_layers=num_of_layers)

    model.compile(
        torch.optim.Adam(model.parameters(), lr=learning_rt),
        torch.nn.MSELoss(),
        metrics=[lighter.metrics.PSNR()],
        device=device
    )

    train_l, test_l = model.fit(
        train_loader,
        epochs=epochs,
        validation_loader=val_loader,
        validation_freq=10,
        callbacks=[
            lighter.callbacks.History(),
            lighter.callbacks.Checkpoint('./checkpoints/DnCNN.pt'),
        ]
    )