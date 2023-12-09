This repository contains the code for two denoising models: Model 1 and Model 2.
Using CNN architecture
For encoder: 3 main layers
For decode: 3 main layers with Sigmoid activation at the end

# Models Overview
| Model | Loss | Random Seed | Description |
|-------|------|--------------|-------------|
| Model 1 | ~0.2 | Not set | Baseline model without a specific random seed. |
| Model 2 | <0.13 | 42 | Utilizes a random seed for reproducibility, achieving a lower loss. |

# Autoencoder Denoising code
[Denoising Autoencoder code](https://github.com/HiepDuong/MNIST--Neural-Network/blob/main/MNIST-Denoising-autoencoder/Denoising-Autoencoder-Model)