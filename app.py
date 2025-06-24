import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# --- Load trained model and preprocessors ---
model = load_model("model1.h5")
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# --- Audio feature extraction ---

def extract_features(data, sample_rate):
    result = np.array([])

    # 1. Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # 2. Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

    # 3. MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # 4. Delta MFCC
    delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    result = np.hstack((result, delta_mfcc))

    # 5. RMS Energy
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # 6. Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    # 7. Spectral Centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, centroid))

    # 8. Spectral Bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, bandwidth))

    # 9. Spectral Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, contrast))

    return result

# --- Streamlit UI ---
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` file to predict the speaker's **emotion**.")

uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    try:
        # Load audio
        data, sr = librosa.load(uploaded_file, duration=2.5, offset=0.6)

        # Extract & preprocess features
        features = extract_features(data, sr).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=2)  # reshape for Conv1D

        # Predict emotion
        prediction = model.predict(features_scaled)
        predicted_label = encoder.inverse_transform(prediction)[0][0]

        # Display result
        st.success(f"🧠 Predicted Emotion: **{predicted_label.upper()}**")

    except Exception as e:
        st.error(f"Error processing the audio: {e}")
