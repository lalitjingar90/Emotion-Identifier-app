
# Emotion Recognition from Speech using Deep Learning

This project implements a deep learning-based system for recognizing emotions from speech signals. It uses audio feature extraction techniques and a Convolutional Neural Network (CNN) model to classify emotions such as happiness, anger, sadness, and more from raw audio inputs. The project is built in Python using `librosa`, `scikit-learn`, and `Keras`.

---

## Project Objectives

* Extract relevant audio features from speech recordings
* Train a deep learning model to classify emotional states
* Evaluate model performance using standard classification metrics
* Explore opportunities for extending the system to real-time applications

---

## Dataset

The system is trained on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset. It contains 7356 files from 24 professional actors, with emotions categorized as:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

Each file encodes metadata including emotion, actor ID, intensity, and modality.

---

## Preprocessing Pipeline

1. **Feature Extraction**
   Audio signals are processed using `librosa` to extract features such as:

   * Mel-frequency cepstral coefficients (MFCCs)
   * Chroma features
   * Spectral contrast
   * Tonnetz
   * Zero-crossing rate
   * Root Mean Square (RMS) energy

2. **Data Normalization**
   All features are standardized using `StandardScaler` to improve convergence during training.

3. **Label Encoding**
   Emotion labels are converted to one-hot encoded vectors.

4. **Train-Test Split**
   The dataset is split into training and testing sets using a 75:25 ratio.

---

## Model Architecture

The model is implemented using the Keras Sequential API and consists of:

* 1D Convolutional layers with ReLU activation
* Batch normalization and dropout layers for regularization
* MaxPooling layers to reduce dimensionality
* Dense layers for final emotion classification

**Training enhancements:**

* ModelCheckpoint to save the best-performing model
* ReduceLROnPlateau for adaptive learning rate adjustment

---

## Evaluation Metrics

The model performance is evaluated using:

* Accuracy score on the test set
* Classification report (precision, recall, F1-score)
* Confusion matrix for visual analysis of predictions

> The system achieves an overall test accuracy above 85%.

---

## How to Run

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the Jupyter Notebook:

```bash
jupyter notebook Emotion_Recognition.ipynb
```

3. Download the RAVDESS dataset and set the dataset path in the notebook:

```python
ravdees_speech = "https://zenodo.org/records/1188976"
```

---

## Future Scope

* Integrate LSTM/GRU layers to capture temporal audio patterns
* Support real-time audio input and prediction
* Experiment with transformer-based audio models

---

## License

This project is shared under the MIT License. You are free to use, modify, and distribute the code with proper attribution.

---

Would you like this written directly into a `README.md` file and added to your repo, or should I generate the file for you to download?
