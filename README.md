# Shake It off - Seismic Detection



## NASA Hackathon: Seismogram Data Processing and Event Detection

This project is part of a NASA hackathon challenge and focuses on processing lunar seismogram data to detect seismic events. The main goals of the project are to filter and segment seismic data, process it for noise reduction, and train a machine learning model to detect seismic events from the processed data. 

## Table of Contents
- [Overview](#overview)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Signal Processing](#signal-processing)
- [Segmenting Seismic Data](#segmenting-seismic-data)
- [Machine Learning Model](#machine-learning-model)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Acknowledgements](#acknowledgements)

---

## Overview

This notebook processes lunar seismic data from the Apollo 12 mission to detect seismic events. The data is pre-processed to filter out noise, then segmented into smaller windows to identify events. The main steps are:

1. **Data Loading**: Load seismic data and the associated catalog containing event timestamps.
2. **Signal Processing**: Apply filtering techniques to remove noise and prepare the data for analysis.
3. **Window Segmentation**: Segment seismic data into smaller windows for feature extraction.
4. **Event Labeling**: Label each window with either an event or no event based on the catalog.
5. **Model Training**: Train a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model to detect events from the segmented windows.
6. **Evaluation**: Evaluate the trained model on validation data and report accuracy.

---

## Data Processing Pipeline

The data comes from the Apollo 12 mission, containing both seismic waveforms (`mseed` files) and event catalogs (`csv` files). The catalog lists the times when seismic events were detected. We pre-process the data and structure it for machine learning.

Steps:

1. **Load Data**: The catalog is loaded into a pandas DataFrame, and seismic data is read from `.mseed` files.
2. **Filter Data**: Bandpass filtering is applied to reduce noise and focus on frequencies of interest (0.6 Hz to 1 Hz).
3. **Wavelet Transformation**: Continuous Wavelet Transform (CWT) is used for signal decomposition (though not currently applied due to convergence issues).

---

## Signal Processing

The seismic data is processed using various methods to clean and enhance the signal:

- **Bandpass Filter**: Removes high-frequency noise and low-frequency drift from the data.
- **Continuous Wavelet Transform (CWT)**: Intended to decompose the signal into wavelet coefficients, but due to instability, the filtered signal is currently used without wavelet reconstruction.
- **Spike Removal**: Outliers (spikes) are detected and optionally interpolated for further noise reduction.
  
The `signal_processing()` function handles filtering and spike correction.

### Output sample of signal processing
![image](https://github.com/user-attachments/assets/d0f91a41-8316-479e-abd1-d4139800c0b8)


---

## Segmenting Seismic Data

To train the machine learning model, the seismic data is divided into windows. Each window represents a time slice of the seismic recording. The process involves:

1. **Sliding Windows**: Each seismogram is split into fixed-sized windows (960 seconds) with a step of 120 seconds.
2. **Normalization**: Each window is normalized using the interquartile range (IQR) to center the data around zero and reduce variance.
3. **Labeling**: Windows that overlap with a seismic event (based on the catalog) are labeled as `1` (event), while non-overlapping windows are labeled as `0` (no event).

The segmentation process ensures that the data is ready for training a model that can differentiate between event and non-event windows.

---

## Machine Learning Model

A deep learning model is implemented using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model architecture includes:

1. **Convolutional Layers**: Extract temporal patterns from the seismic data.
2. **LSTM Layer**: Capture sequential dependencies in the data.
3. **Dense Output Layer**: Output a probability of a seismic event using a sigmoid activation function.

The model is trained using binary cross-entropy loss and optimized with the Adam optimizer.

```python
model = Sequential()
model.add(Input(shape=(Xfinal.shape[1], 1)))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

- **Class Balancing**: Due to the imbalance in the number of event vs. non-event windows, class weights are computed to ensure the model doesn't favor the majority class (no event).
  
---

## Requirements

To run this project, the following packages are required:

```bash
numpy
pandas
matplotlib
scipy
pywt
obspy
tensorflow
scikit-learn
```

---

## How to Run

1. **Install Dependencies**: Install the required Python packages listed above.
2. **Prepare Data**: Ensure the seismic data (`mseed` files) and catalog CSV files are in the appropriate directories.
3. **Run the Notebook**: Execute the notebook step by step. The key steps include:
    - Loading the seismic data and catalog.
    - Processing the seismic signals.
    - Segmenting the data.
    - Training the machine learning model.
4. **Model Evaluation**: Evaluate the model on the validation set to measure its accuracy in detecting seismic events.

---

## Acknowledgements

This project is developed as part of a NASA hackathon challenge. Special thanks to the Apollo missions for providing lunar seismic data, which made this project possible.
