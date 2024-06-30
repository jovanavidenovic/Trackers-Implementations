# Foundational object tracking algorithms

This repository contains implementations of fundamental object tracking algorithms in Python.
The original PyTorch implementation of the SiamFC tracker has been updated to a long-term variant.

## Repository Structure
The following is an overview of the repository structure.
```bash
├── DCF/                # MOSSE tracker (discriminative correlation filter)
├── mean_shift/         # Mean-shift tracker
├── optical_flow/       # Lucas-Kanade and Horn-Schunck optical flow
├── particle_filter/    # Particle filter tracking, Kalman filter
└── siamfc/             # Updated SiamFC tracker to long-term variant
```

