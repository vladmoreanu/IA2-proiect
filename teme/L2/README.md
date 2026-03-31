## Homework 🔬 (20 pts, teams of max. 3)

Analyse the influence of additive noise on blind source separation. Given a mixture signal for which the true sources are known, your goal is to evaluate how well a separation model performs when the input mixture is corrupted by Gaussian noise $z \sim \mathcal{N}(0, \sigma^2)$, for varying noise levels $\sigma$.

### Tasks:

**1. Choose noise levels (`4 pts`)**

Select three distinct values of $\sigma$ such that the SNR between the clean mixture $x$ and the noisy mixture $x + z$ satisfies:

$$\text{SNR}(x,\, x+z) \leq 20\ \text{dB}$$

For each chosen $\sigma$, report the corresponding SNR (in dB). Make sure the three values span a meaningful range (e.g., low, medium, and high noise).

> 💡 Recall: $\text{SNR} = 10 \log_{10}\left(\frac{\|x\|^2}{\|z\|^2}\right)$

**2. Evaluate the original model under noise (`8 pts`)**

For each of the three $\sigma$ values from Task 1, corrupt the **test set** mixtures with noise sampled from $\mathcal{N}(0, \sigma^2)$ and compute the **SI-SNR-PIT** of the original (clean-trained) model.

- Report all results in the table below *(3 pts)*
- Analyse the trend: how does increasing noise degrade separation performance? Are the results consistent with your expectations based on the SNR values? *(5 pts)*

**3. Train a noise-robust model (`8 pts`)**

Pick **one** of the three $\sigma$ values from Task 1. Train a new model on mixtures perturbed with noise sampled from $\mathcal{N}(0, \sigma^2)$ during training.

- Evaluate this new model on the **test set** under all three noise conditions and add the results to the table *(3 pts)*
- Compare and analyse: does training with noise improve robustness? Under which conditions does it help most or least? *(5 pts)*

### Results Table

Your final table should look like this:

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
    <td>…</td>
    <td>…</td>
    <td>…</td>
  </tr>
  <tr>
    <td>Model trained on σ = …</td>
    <td>…</td>
    <td>…</td>
    <td>…</td>
  </tr>
</tbody>
</table>
