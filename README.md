# Radiative Cooling Surrogate Model

Neural network surrogate for radiative cooling in astrophysics simulations, built during the Intro to HPC 2026 bootcamp (Argonne, NERSC, Oak Ridge). This is my copy of Project04 from the bootcamp repo.

## Overview

Chemistry and cooling solvers are a major cost in astrophysics simulations. This project trains a neural network to approximate the cooling calculation, so it runs much faster than the full solver. The workflow goes from data generation, to training, to scaling across GPUs on HPC systems.

## What's in here

- `01_perlmutter-installation.md`: environment setup on Perlmutter
- `02_nb1_chemistry_solver.ipynb`: chemistry solver and training data generation
- `03_nb2_surrogate_training.ipynb`: surrogate model training
- `04_nb3_gpu_scaling.ipynb`: GPU scaling experiments
- `05_train_ddp.py`: multi-GPU training with PyTorch DDP
- `Containerfile`, `build-podman-container.sh`, `build-shifter-container.sh`: containerized environment

## Systems

Perlmutter (NERSC), Aurora (Argonne)

## Credits

Project led by Snigdaa S. Sethuram and Luis Rangel DaCosta. Original bootcamp repo: [Intro-HPC-2026](https://github.com/ORG/Intro-HPC-2026)

---
