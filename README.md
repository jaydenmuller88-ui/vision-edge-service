# Vision Edge Service — Real-Time YOLO Hazard Detection [INCOMPLETE / PAUSED]

A real-time computer vision pipeline built with Python and Ultralytics YOLO, optimized for local NVIDIA RTX GPU inference to monitor industrial safety, identify hazards, and track warehouse compliance.

> **Project Status:** **Incomplete / Paused**. Development was paused after encountering significant object recognition and multi-frame tracking issues in cluttered warehouse footage (such as bounding-box jitter, ID switching, and false positives under varying lighting), which require further dataset expansion and tracking parameter tuning.

## Core Features

* **Edge GPU Inference:** Leverages CUDA acceleration for real-time object detection and bounding-box rendering across video streams.
* **Custom Model Training:** Includes dedicated training scripts to fine-tune YOLO weights on domain-specific warehouse hazard datasets.
* **Validation & Diagnostics:** Built-in utilities to verify CUDA/GPU availability and benchmark detection accuracy against recorded warehouse video feeds.

## Tech Stack

* **Computer Vision & Deep Learning:** `Ultralytics YOLO`, `PyTorch`, `OpenCV`
* **Data Processing:** `NumPy`

## Repository Structure

```text
vision-edge-service/
├── app.py              # Main real-time inference and detection entry point
├── train.py            # Custom YOLO model training script
├── test_video.py       # Video feed evaluation and tracking benchmark script
├── test_gpu.py         # CUDA and GPU hardware verification utility
├── .gitignore          # Excludes heavy datasets, model weights, and video files
└── README.md
