import torch
import torchaudio
from hw_utils import LibriMixDataset
from convtasnet import ConvTasNet
import matplotlib.pyplot as plt
import toml
import librosa



model_path = 'teme/L2/models/ConvTasNet_Noise.pth'
sample_key = 1

val_dataset = LibriMixDataset(
    subset="val",
    typ='sigma0',
    root='_fit/hw2/data/mini_libri2mix',
    segment_length=8,
    sample_rate=8000
)

test = val_dataset[sample_key][0]
test = torch.from_numpy(test)
test = test.unsqueeze(0)

val_dataset.listen_samples(sample_key)

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
        signal = channel_data.numpy()

        plt.figure(figsize=(10, 1))
        librosa.display.waveshow(signal, sr=8000)
        plt.title(f'Predicted_Source_{i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.ylim([-1, 1])
        plt.show()

        torchaudio.save(f'source_{i}.wav', channel_data, sample_rate)
