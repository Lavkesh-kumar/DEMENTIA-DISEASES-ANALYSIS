# Dementia Diseases Analysis via Clock Drawing Test (CDT)

## Background
Dementia is one of the most common neurocognitive syndromes in the world, in which memory and the ability to perform daily life activities deteriorate and get worse. If detected early, it can help in better management of the disease and can save a person from a lifelong dependency on severe medication.

## Clock Drawing Test (CDT)
The CDT is a proven preliminary test for the detection of neurocognitive diseases like dementia. The patient is given a pen and paper and is asked to draw an analog clock showing the time "11:10". Based on how the patient draws, a medical practitioner can detect the chances of the patient suffering from dementia.

## Goal of the Project
The core objective is to digitize the process of CDT by using deep learning models (specifically an image classification model built on ResNet18 via `timm`) utilizing the dataset of hand-drawn clock images. 

**Severity Scale:**
The severity of the disease is evaluated on a scale of **0-5**, with the following specific classifications:
*   `0`: Severe Dementia (Worst Case)
*   `1`: Moderate to Severe Dementia
*   `2`: Moderate Dementia
*   `3`: Mild Dementia
*   `4`: Very Mild Dementia
*   `5`: Normal (No signs of Dementia)

## Architecture Overview
This repository contains:
1.  **`Analysis_model.ipynb`**: Original PyTorch & Jupyter code for training the model using Albumentations for augmentation.
2.  **`app.py`**: A Flask web application that serves the model globally, handling image uploads and returning the severity prediction instantly.
3.  **`utils/inference.py`**: The backend utility that loads the `.pth` weights, transforms raw images using Albumentations, and extracts predictions.
4.  **`templates/index.html`**: A modern, dark-mode user interface designed for easy Drag & Drop clinical usage.

## Setup & Running the Interface

### Requirements
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### Running the App
Start the Flask web server:
```bash
python app.py
```
Open a browser and navigate to `http://localhost:5000` to interact with the Dementia Analyzer.

## Evaluation Metric
During training, the evaluation metric used for the models is **Classification Accuracy**.
`Accuracy_score = Number of correct predictions / Total number of predictions`

