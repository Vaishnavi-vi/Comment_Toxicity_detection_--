import streamlit as st
import tensorflow as tf #type:ignore 
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
from PIL import Image
import pickle

# Load model and tokenizer
model = tf.keras.models.load_model("toxic_cnn_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
# Config
max_len = 100 

def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

def predict(text):
    processed = preprocess(text)
    prob = model.predict(processed)[0][0]
    label = "Toxic" if prob > 0.5 else "Non-Toxic"
    return label, prob
    
    
page=st.sidebar.radio("Go to",["Home","CNN Toxic Comment Classifier"])
if page=="Home":
    st.header("Toxic Comment Classifier")
    image=Image.open("C:\\Users\\Dell\\OneDrive - Havells\\Downloads\\Toxic_comment.png")
    st.image(image,use_container_width=True)
elif page=="CNN Toxic Comment Classifier":
    st.title("CNN Toxic Comment Classifier")
    text_input = st.text_area("Enter a comment")
    if st.button("Predict"):
        if text_input.strip() == "":
            st.warning("Please enter a comment.")
        else:
            label, prob = predict(text_input)
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {prob:.2f}")