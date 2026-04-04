import torch
import torchaudio
from hw_utils import LibriMixDataset
from convtasnet import ConvTasNet
import toml


model_path = 'teme/L2/models/ConvTasNet_Baseline.pth'

val_dataset = LibriMixDataset(
    subset="val",
    typ='both',
    root='_fit/hw2/data/mini_libri2mix',
    segment_length=8,
    sample_rate=8000
)

test = val_dataset[0][0]
test = torch.from_numpy(test)
test = test.unsqueeze(0)

config_path = 'teme/L2/noisy_ss.toml'

with open(config_path, 'r') as f:
    config = toml.load(f)

model = ConvTasNet(**config['model'])

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()
output = model(test)

output = output.squeeze(0)

sample_rate = 8000

with torch.no_grad():
    for i in range(output.shape[0]):
        channel_data = output[i:i+1, :]
        torchaudio.save(f'source_{i}.wav', channel_data, sample_rate)
