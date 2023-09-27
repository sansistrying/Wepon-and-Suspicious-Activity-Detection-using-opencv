# Security Monitoring System with Face Emotion Recognition and Weapon Detection

## Overview

This Python script demonstrates a security monitoring system that uses computer vision techniques to detect and respond to potential security threats. The system combines face emotion recognition and weapon detection to enhance security in a given environment.

## Features

- **Face Emotion Recognition:** The system uses a pre-trained deep learning model to recognize emotions on human faces captured by a camera feed. It can detect emotions such as anger, fear, happiness, sadness, surprise, and more.

- **Weapon Detection:** The system employs YOLO (You Only Look Once) object detection to identify weapons within the camera feed. When a weapon is detected, it raises an alarm.

- **Suspicious Activity Detection:** The system can identify suspicious activities based on detected emotions and weapon presence. It raises a flag for suspicious activity if certain conditions are met.

## Prerequisites

Before running the script, make sure you have the following prerequisites:

- Python 3.x
- OpenCV (cv2) library
- NumPy library
- TensorFlow library (for the emotion recognition model)
- A trained emotion recognition model (in the example, "FacialExpressionModel.h5" is used)
- YOLO weights and configuration files (for weapon detection)

You can download the trained emotion recognition model from this [Google Drive link](https://drive.google.com/drive/folders/1DtSixBhCt3Ac2IxRnLDiaWtLzimh7MGi?usp=sharing).

- YOLO weights and configuration files for weapon detection

You can obtain the YOLO configuration files and weights from this [Google Drive link](https://drive.google.com/drive/folders/1DtSixBhCt3Ac2IxRnLDiaWtLzimh7MGi?usp=sharing).


## Usage

### 1. Install Dependencies

Install the necessary Python libraries using `pip`:

```bash
pip install opencv-python numpy tensorflow
