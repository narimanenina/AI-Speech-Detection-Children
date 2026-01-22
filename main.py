import numpy as np
import scipy.io.wavfile as wav
import tempfile
import tensorflow as tf
import librosa
import streamlit as st
import sounddevice as sd
import json
from librosa.feature import mfcc


# Open the JSON file
with open('params.json', 'r') as f:
    # Load the JSON data into a dictionary
    params = json.load(f)

# Access the variables
max_len = params['max_len']
classes = params['classes']
n_mfcc = params['n_mfcc']
# تحميل النموذج المدرب وإعادة تجميعه
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cnn_Speech.hdf5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_model()
# وظيفة لتسجيل الصوت
def record(duration=5, fs=22050):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # الانتظار حتى انتهاء التسجيل
    return recording, fs

# وظيفة لحفظ الصوت
def save_audio(recording, fs):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        wav.write(f.name, fs, recording)
        return f.name

def find_class(dict, value):
    for key, val in dict.items():
        if val == value:
            return key
    return None

def predict(audio_data):
    """
    Preprocesses audio data, extracts MFCC features, and predicts speech disorder.

    Args:
        audio_data (np.ndarray): Raw audio data from the user recording.

    Returns:
        str: Predicted label ("normal" or "disorder")
    """
    # Preprocess audio data
    mfcc_features = mfcc(y=audio_data, sr=22050, n_mfcc=n_mfcc)
    if mfcc_features is None:
        return "Error extracting features"

    # Assuming your model expects a specific sequence length (max_len from training)
    # Pad the features with zeros to match the maximum length
    if mfcc_features.shape[1] < max_len:
        pad_width = ((0, 0), (0, max_len - mfcc_features.shape[1]))
        mfcc_features = np.pad(mfcc_features, pad_width=pad_width, mode='constant')
    else:
        mfcc_features = mfcc_features[:, :max_len]

    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add extra dimension for batch processing
    prediction = model.predict(mfcc_features)[0]
    predicted_label = np.argmax(prediction)
    return find_class(classes, predicted_label)

# تطبيق Streamlit
st.title("Welcome to a website for detecting speech disorders in elementary school children")

# زر بدء التسجيل
if st.button("Start Registration"):
    recording, fs = record()
    audio_data = save_audio(recording, fs)
    st.success("Registration is complete. Click on 'Predict' to get the result.")
    st.session_state.audio_data = audio_data  # حفظ المسار في حالة الجلسة

# تحميل ملف صوتي عن طريق السحب والإسقاط
uploaded_file = st.file_uploader("Drag and drop the audio file here or click to choose a file", type=["wav"])

if uploaded_file is not None:
    # حفظ الملف الصوتي في ملف مؤقت
    audio_data = tempfile.mktemp(suffix='.wav')
    with open(audio_data, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success("Audio file upload. Click on predict to get the result.")
    st.session_state.audio_data = audio_data  # حفظ المسار في حالة الجلسة

# زر تشغيل الصوت
if st.button("Audio playback"):
    if 'audio_data' in st.session_state:
        audio_data = st.session_state.audio_data
        st.audio(audio_data, format="audio/wav")
    else:
        st.error("Please record the audio or upload the audio file first.")

# زر التنبؤ
if st.button("prediction"):
    if 'audio_data' in st.session_state:
        # Process the uploaded audio
        st.write("Processing...")
        try:
            audio_data, sr = librosa.load(st.session_state.audio_data, sr=22050)  # Load the correct audio data

            # Make prediction using the defined function
            prediction = predict(audio_data)
            st.write(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"An error occurred while processing the audio: {e}")
    else:
        st.error("Please record the audio or upload the audio file first.")



