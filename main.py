import numpy as np
import scipy.io.wavfile as wav
import tempfile
import tensorflow as tf
import librosa
import streamlit as st
import sounddevice as sd
import json
import os
from librosa.feature import mfcc

# 1. تحميل الإعدادات مع معالجة الأخطاء
try:
    with open('params.json', 'r') as f:
        params = json.load(f)
    max_len = params['max_len']
    classes = params['classes']
    n_mfcc = params['n_mfcc']
except FileNotFoundError:
    st.error("ملف params.json غير موجود!")
    st.stop()

# 2. تحميل النموذج
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('speech_model.keras')
        # لا حاجة لعمل compile إذا كان النموذج محفوظاً بكامل إعداداته، لكن لا بأس بها
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {e}")
        return None

model = load_model()

# 3. وظائف معالجة الصوت
def record(duration=5, fs=22050):
    st.info(f"جاري التسجيل لمدة {duration} ثوانٍ...")
    # ملاحظة: sd.rec قد لا تعمل على Streamlit Cloud لأنها تتطلب الوصول لعتاد السيرفر
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording, fs

def save_audio(recording, fs):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wav.write(temp_file.name, fs, (recording * 32767).astype(np.int16)) # تحويل لـ 16-bit PCM
    return temp_file.name

def find_class(class_dict, value):
    for key, val in class_dict.items():
        if int(val) == int(value): # التأكد من مقارنة الأرقام بشكل صحيح
            return key
    return "Unknown"

def predict(audio_path):
    # تحميل الصوت باستخدام librosa للتأكد من التنسيق الصحيح
    audio_data, sr = librosa.load(audio_path, sr=22050)
    
    # استخراج ميزات MFCC
    mfcc_features = mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    
    # ضبط الطول (Padding/Truncating)
    if mfcc_features.shape[1] < max_len:
        pad_width = max_len - mfcc_features.shape[1]
        mfcc_features = np.pad(mfcc_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_features = mfcc_features[:, :max_len]

    # إضافة الأبعاد المطلوبة للنموذج (Batch, Height, Width, Channels) 
    # يعتمد هذا على شكل input_shape في نموذجك (مثلاً: 1, 40, 100, 1)
    mfcc_features = mfcc_features[np.newaxis, ..., np.newaxis] 
    
    prediction = model.predict(mfcc_features)
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    return find_class(classes, predicted_label)

# 4. واجهة Streamlit
st.title("الكشف عن اضطرابات النطق لدى الأطفال")

# استخدام Session State لتخزين مسار الملف
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None

col1, col2 = st.columns(2)

with col1:
    if st.button("بدء التسجيل المباشر"):
        try:
            recording, fs = record()
            path = save_audio(recording, fs)
            st.session_state.audio_path = path
            st.success("تم التسجيل بنجاح!")
        except Exception as e:
            st.error(f"خطأ في الميكروفون: {e}")

with col2:
    uploaded_file = st.file_uploader("أو ارفع ملف صوتي", type=["wav"])
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(uploaded_file.getbuffer())
        st.session_state.audio_path = temp_file.name
        st.success("تم رفع الملف!")

# عرض الخيارات بعد توفر ملف صوتي
if st.session_state.audio_path:
    st.audio(st.session_state.audio_path)
    
    if st.button("تحليل النتيجة (Predict)"):
        with st.spinner("جاري التحليل..."):
            result = predict(st.session_state.audio_path)
            st.subheader(f"النتيجة المتوقعة: {result}")

