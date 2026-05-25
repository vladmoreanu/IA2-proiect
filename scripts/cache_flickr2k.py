import torch
from datasets import Flickr2K

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
dataset = Flickr2K("tiled_pairs", device=device)

print(len(dataset))
