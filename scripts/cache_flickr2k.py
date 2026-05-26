import torch
import torch.utils.data as torchdata
from datasets import Flickr2K

from multiprocessing import freeze_support

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("System :")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version PyTorch expects: {torch.version.cuda}")
    print(f"  Using device: {device}")

    device = torch.device(device)
    dataset = Flickr2K("tiled_pairs", device=device)

    print(len(dataset))

    # ###

    # subset = torchdata.Subset(dataset, range(100))

    # dataloader = torchdata.DataLoader(
    #     subset,
    #     batch_size=5
    # )
    # print(len(subset))
    # print(len(dataloader))

if __name__ == "__main__":
    freeze_support()
    main()
