# Mammographic Density Prediction (BI-RADS Classification)

*A Deep Learning project developed for the Machine Learning for Imaging (COMP70014) coursework at Imperial College London.*

## Project Overview
This repository contains a deep learning pipeline designed to automatically predict breast tissue density from digital screening mammography data. Breast tissue density is a critical risk factor in breast cancer and is categorised into four classes according to the **BI-RADS** (Breast Imaging Reporting and Data System) standard (Classes A through D).

The goal of this project goes beyond standard classification; it emphasises model interpretability, clinically sound data augmentation, and subgroup performance analysis to understand how domain shifts (such as the presence of breast implants) affect model reliability.

## Key Features & Methodology

* **Clinically-Informed Data Augmentation:** Implemented a two-stage augmentation pipeline (Photometric and Geometric) using conservative ranges (e.g., restricted rotation and affine transformations) to ensure clinically relevant density patterns were not artificially distorted.
* **Subgroup Performance Analysis:** Evaluated the model across specific patient subgroups, assessing variance in accuracy based on laterality (Left/Right) and the presence of implants.
* **Feature Embedding Inspection:** Analysed the high-dimensional representations learned by the model. Used **PCA** and **t-SNE** dimensionality reduction techniques to project embeddings and visually inspect class separability, domain shifts, and the model's understanding of the continuous density spectrum.

## Repository Structure

* `notebook_final.ipynb`: The core codebase containing the PyTorch implementation of the dataset loaders, augmentation pipelines, model architecture, training loops, and the t-SNE/PCA evaluation scripts.
* `Report.pdf`: A comprehensive academic report detailing the experimental design, justification for the augmentation strategies, and visual analysis of the model's feature embeddings. 

## Technologies Used
* **Python**
* **PyTorch** (Deep Learning Architecture & DataLoaders)
* **Scikit-Learn** (PCA, t-SNE)
* **Torchvision** (Image Augmentation)