# Spatio-Temporal Features for Action Recognition :

## Overview

This project explores the field of action recognition by analyzing the efficacy of local spatio-temporal features in classifying human actions in video sequences. Utilizing the UCF Sports Action Dataset, we aim to ascertain the classification accuracy by integrating appearance and motion data.

---

## System Requirements

- **Python 3.6** or higher
- **TensorFlow 1.5** (or above) - GPU version recommended for enhanced performance
- **OpenCV 3.0** (or above)
- **Numpy**

## Installation Guide

To set up your environment for the project, follow these steps:

1. **Install Required Libraries**:
   Ensure Python, TensorFlow, OpenCV, and Numpy are installed on your system. You can install them using pip:

   ```bash
   pip install tensorflow
   pip install opencv-python
   pip install numpy
   ```

2. **Set Up Project Directory**:
   Create a directory specifically for this project to keep all related files organized.

3. **Download and Set Up TensorFlow Model**:
   - Download `faster_rcnn_inception_v2_coco.tar.gz` from the [TensorFlow Detection Model Zoo](https://tensorflow.org/models).
   - Extract the contents into your project directory.
   
4. **Get the Detection Script**:
   - Download `tensorflow-human-detection.py` [from this repository](#). Replace the placeholder paths in the script with the correct paths to the extracted model and your input videos.

## Configuration

Before running the scripts, configure the paths in your scripts:

- **Model Path**: Path to the TensorFlow model, typically ending with `frozen_inference_graph.pb`.
- **Video Path**: Directory path where your input videos are stored.
- **CSV Path**: Path where descriptors will be saved in a CSV file.

## Procedure

1. **Prepare Input Data**:
   - Ensure that your videos are placed in the specified `Video Path`.
   - Edit `descriptor_saving(final).py` with paths pointing to your model, video, and output CSV paths.

2. **Run the Descriptor Saving Script**:
   ```bash
   python descriptor_saving(final).py
   ```
   This will process each video and save the extracted descriptors into a CSV file named after the input video.

3. **Handling Multiple Videos**:
   If you need to process multiple videos, you can use `multiple.py` to automate the processing of multiple files in a directory.

## Training the Machine Learning Model

1. **Prepare Training and Test Sets**:
   - Ensure you have `train.txt` and `test.txt` files that list the video paths and their corresponding labels for training and testing.

2. **Train Your Model**:
   Use the extracted descriptors to train your machine learning model. Adjust the training script to read from your generated CSV files.

## Output

- The descriptor data (HOG and HOF features) for each video is saved in the designated CSV file, facilitating easy use in machine learning models for action recognition.

---

## Notes

- Modify paths directly in the scripts where placeholders like `/path/to/...` are used.
- Ensure consistency in data formats and paths used throughout the project for smooth operation.

This README provides a comprehensive guide to setting up and running the spatio-temporal action recognition project. For elaborate details, refer to the in-code comments and linked resources.  
