# %% [markdown]
## Homework 🔬 (20 pts, teams of max. 3)
# Analyse the influence of additive noise on blind source separation. Given a
# mixture signal for which the true sources are known, your goal is to evaluate
# how well a separation model performs when the input mixture is corrupted by
# Gaussian noise $z \sim \mathcal{N}(0, \sigma^2)$, for varying noise
# levels $\sigma$.

# %% [markdown]

# **0. Loading Baseline Model and Prepping Dataset**

# %%
from convtasnet import ConvTasNet, SI_SNR_PIT
from hw_utils import prep_dataset, LibriMixDataset

import lighter

import torch
from torch.utils.data import DataLoader

import os
import toml

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_path = './_fit/hw2/noisy_ss.toml'

with open(config_path, 'r') as f:
    config = toml.load(f)

config_dir, _ = os.path.split(config_path)
working_dir = config.get('working_dir', config_dir)

prep_dataset(os.path.join(
    working_dir, config['dataset'].get('root')
))

model = ConvTasNet(**config['model'])

model.compile(
    torch.optim.Adam(model.parameters(), **config['optimizer']),
    SI_SNR_PIT(),
    metrics=[],
    device=device,
)

model.load(os.path.join(
    working_dir, 'models/ConvTasNet_baseline.pth'
))

# %% [markdown]

# **1. Choose noise levels (`4 pts`)**

# Select three distinct values of $\sigma$ such that the SNR between the clean
# mixture $x$ and the noisy mixture $x + z$ satisfies:

# $$\text{SNR}(x,\, x+z) \leq 20\ \text{dB}$$

# For each chosen $\sigma$, report the corresponding SNR (in dB). Make sure the
# three values span a meaningful range (e.g., low, medium, and high noise).

# > 💡 Recall: $\text{SNR} = 10 \log_{10}\left(\frac{\|x\|^2}{\|z\|^2}\right)$

# %%

# TODO: NOISE ADDITION HERE

import numpy as np

noise = np.random.normal(0, noise_sigma, original_img.shape).astype(np.float32)  # noise


# %% [markdown]

# **2. Evaluate the original model under noise (`8 pts`)**

# For each of the three $\sigma$ values from Task 1, corrupt the **test set** mixtures with noise sampled from $\mathcal{N}(0, \sigma^2)$ and compute the **SI-SNR-PIT** of the original (clean-trained) model.

# - Report all results in the table below *(3 pts)*
# - Analyse the trend: how does increasing noise degrade separation performance? Are the results consistent with your expectations based on the SNR values? *(5 pts)*

# %%

results = {
    'model'     : [],
    'l_sigma0'  : [],
    'l_sigma1'  : [],
    'l_sigma2'  : [],
}

results['model'].append('baseline')
for idx in range(3):
    val_dataset = LibriMixDataset(
        subset="val",
        typ='sigma' + idx,
        **config['dataset'],
    )

    val_loader   = DataLoader(  val_dataset, **config['dataloader'])

    val_l = model.evaluate(
        val_loader,
        callbacks = [
            lighter.callbacks.History(),
        ],
    )

    results['l_sigma'+idx].append(val_l)

# %% [markdown]


# %% [markdown]
# **3. Train a noise-robust model (`8 pts`)**

# Pick **one** of the three $\sigma$ values from Task 1. Train a new model on mixtures perturbed with noise sampled from $\mathcal{N}(0, \sigma^2)$ during training.

# - Evaluate this new model on the **test set** under all three noise conditions and add the results to the table *(3 pts)*
# - Compare and analyse: does training with noise improve robustness? Under which conditions does it help most or least? *(5 pts)*

# %% [markdown]
# **3.1. Training**

model_path = os.path.join(
    working_dir, 'models/ConvTasNet_Noise.pt'
)

if os.path.exists(model_path):
    model.load(model_path)
else:
    train_dataset = LibriMixDataset(
        subset="train",
        typ='sigma1',
        **config['dataset'],
    )

    val_dataset = LibriMixDataset(
        subset="val",
        typ='sigma1',
        **config['dataset'],
    )

    train_loader = DataLoader(train_dataset, **config['dataloader'])
    val_loader   = DataLoader(  val_dataset, **config['dataloader'])

    chkpoint_path = os.path.join(
        working_dir, 'models/checkpoints/ConvTasNet_Noise.pt'
    )
    log_path = os.path.join(
        working_dir, 'logs/ConvTasNet_Noise_logs.csv'
    )

    train_l, val_l = model.fit(
        train_loader,
        validation_loader=val_loader,
        callbacks=[
            lighter.callbacks.CSVLogger(log_path),
            lighter.callbacks.History(),
            lighter.callbacks.Checkpoint(
                chkpoint_path,
                save_best_only=True
            ),
        ],
        **config['fit'],
    )

# %% [markdown]
# **3.2. Evaluation**

results['model'].append('trained')
for idx in range(3):
    val_dataset = LibriMixDataset(
        subset="val",
        typ='sigma' + idx,
        **config['dataset'],
    )

    val_loader   = DataLoader(  val_dataset, **config['dataloader'])

    val_l = model.evaluate(
        val_loader,
        callbacks = [
            lighter.callbacks.History(),
        ],
    )

    results['l_sigma'+idx].append(val_l)

# %% [markdown]
# **Final table**

from IPython.display import Markdown, display

# results = {
#     'model'     : ['dummy', 'dummy'],
#     'l_sigma0'  : ['dummy', 'dummy'],
#     'l_sigma1'  : ['dummy', 'dummy'],
#     'l_sigma2'  : ['dummy', 'dummy'],
# }

table = f'''\
<table style="margin: 0px auto;">
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">SI-SNR-PIT on test set — mixtures perturbed with:</th>
  </tr>
  <tr>
    <th>σ = … (SNR ≈ … dB)</th>
    <th>σ = … (SNR ≈ … dB)</th>
    <th>σ = … (SNR ≈ … dB)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Original model (trained on clean mixtures)</td>
    <td>{results['l_sigma0'][0]}</td>
    <td>{results['l_sigma1'][0]}</td>
    <td>{results['l_sigma2'][0]}</td>
  </tr>
  <tr>
    <td>Model trained on σ = …</td>
    <td>{results['l_sigma0'][1]}</td>
    <td>{results['l_sigma1'][1]}</td>
    <td>{results['l_sigma2'][1]}</td>
  </tr>
</tbody>
</table>
'''

display(Markdown(table))
