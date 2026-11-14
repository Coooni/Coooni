# Bitgoeul (Chloe) Kim

**PhD Candidate, Computer Engineering @ Iowa State University** · Computer Vision · Multimodal Foundation Models · Self-Supervised Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bitgoeul-chloekimb24a6b217)
[![Email](https://img.shields.io/badge/Email-goeul8604%40gmail.com-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:goeul8604@gmail.com)

I build and adapt vision foundation models for Earth observation — pretraining multimodal masked autoencoders at scale, cross-modal generation with diffusion models, and downstream prediction from satellite and UAV imagery. Graduating Fall 2027; seeking Summer 2027 research internships in computer vision.

## Research

- **CropMAE — multimodal self-supervised pretraining.** Masked autoencoder pretrained on 8M+ Earth-observation samples (Sentinel-1/2 + weather, soil, elevation; 2017–2024). Multi-GPU PyTorch DDP on Slurm; multi-year, multi-state pretraining beats single-year baselines on held-out 2025 data.
- **Cross-modal diffusion inpainting.** Conditional diffusion model reconstructing missing Sentinel-1 SAR imagery from Sentinel-2 optical inputs.
- **Foundation model adaptation.** Frozen vs. fine-tuned [TerraMind](https://github.com/Coooni/terramind) representations for field-level yield prediction — corn test R² 0.679 (MAE 12.22 bu/acre) on ~600 fields × 24 weekly bins.- **UAV time-series vision.** CNNs on 22,043 soybean plots for maturity classification; 84.9% four-class accuracy, and ~same accuracy from 3 of 8 flights.
## Publications

- **Kim, B.**\*, Blair, S., Jubery, Z., Sarkar, S., Singh, A. (2025). Soybean maturity prediction using two-dimensional contour plots from drone-based time series imagery. *The Plant Phenome Journal*. [doi:10.1002/ppj2.70036](https://doi.org/10.1002/ppj2.70036) — \*co-first author
- Powadi, A., **Kim, B.**, et al. (2026). Geospatial Multimodal Fusion for Field-Level Yield Prediction. *Under review.*
- **Kim, B.**, et al. Satellite Foundation Models for Soybean Trait Prediction. *In preparation.*

## Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazonwebservices&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black)

Distributed training (PyTorch DDP, Slurm, multi-GPU HPC) · Google Earth Engine · C/C++ · SQL
