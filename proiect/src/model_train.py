from model import DnCNN
from dataset import ImageDataset
from torch.utils.data import DataLoader, random_split
import torch
import lighter
from tqdm import tqdm

path_clean = r"./DATASETS/Flickr2K/normal_images"
path_noisy = r"./DATASETS/Flickr2K/noise_images"

batch_size = 16
train_split = 0.8
num_of_layers = 17 # 9  # default 17
epochs = 20
learning_rt = 1e-3
patch_size = 128 # 16  # default 128

num_workers = 4
prefetch_factor = 6

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version PyTorch expects: {torch.version.cuda}")


class History(lighter.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.pbar = tqdm(total=len(self._model.train_loader),
                         desc=f"Epoch {epoch + 1}", unit="batch")

    def on_train_batch_end(self, batch, logs=None):
        self.pbar.update(1)
        if logs:
            self.pbar.set_postfix(loss=f"{logs['train_loss']:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()
        out_str = '  Summary:'
        for k, v in logs.items():
            v = f'{v:.3f}' if v > 0.01 else f'{v:.2e}'
            out_str += ' {:s}={:s}'.format(k, v)
        print(out_str)


dataset = ImageDataset(noisy_dir=path_noisy, clean_dir=path_clean, patch_size=patch_size)

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
    pin_memory=True
    )
val_loader   = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    pin_memory_device=device,
    pin_memory=True
    )

model = DnCNN(num_of_layers=num_of_layers)

model.train_loader = train_loader

model.compile(
    torch.optim.Adam(model.parameters(), lr=learning_rt),
    torch.nn.MSELoss(),
    metrics=[],
    device=device
)

train_l, test_l = model.fit(
    train_loader,
    epochs=epochs,
    validation_loader=val_loader,
    validation_freq=10,
    callbacks=[
        History(),
    ]
)