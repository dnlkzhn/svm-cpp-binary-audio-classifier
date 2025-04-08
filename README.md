# SVM-Based Music Genre Binary Classifier

## Project Overview

The goal of this project is to build a complete **binary classifier** that distinguishes between two music genres based on audio features.  
The Support Vector Machine (SVM) algorithm was **implemented from scratch in C++** to handle the classification task.  
The project covers the full machine learning pipeline, including feature extraction, model training, prediction, and user interaction through a graphical interface.

---

## Resources and Methods Used

- **Dataset**:  
  A reduced version of the **GTZAN music genre dataset** was used, containing audio samples labeled with two genres for binary classification.

- **Data Preprocessing and Analysis**:
  - Exploratory Data Analysis (EDA) was performed, including feature distribution visualization and feature correlation exploration.
  - Feature values were normalized using **Min-Max Scaling** to the [0, 1] range.
  - **NumPy** and **Pandas** were used for data handling and analysis.

- **Model Implementation**:
  - The **SVM classifier** was implemented manually in core **C++**, without the use of external machine learning libraries.

- **Prediction and Audio Processing**:
  - A **Python prediction script** was created for model inference.
  - **Librosa** was used to extract audio features from `.wav` files for prediction.

- **User Interface**:
  - A simple GUI was developed using **CustomTkinter** to allow users to select an audio file and receive classification results.

---

## Results Achieved

- The implemented SVM classifier achieves:
  - **Accuracy**: 97.5%
  - **Precision**: 1.00
  - **Recall**: 0.95
  - **F1-Score**: 0.974
  on the testing subset of the dataset.

- **Additional Deliverables**:
  - A **detailed project report** was prepared, covering:
    - SVM theoretical background.
    - Project goals and a detailed flowchart.
    - Experimental results and evaluation metrics.
    - Description of the implemented user interface.

The project demonstrates that a custom C++ implementation of SVM can achieve high accuracy on real-world data and can be effectively integrated into a user-friendly application.

---
