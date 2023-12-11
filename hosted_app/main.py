import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import altair

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


st.markdown("""
<style>
/* Gradient background */
body {
    background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
}

/* Rounded corners for images */
img {
    border-radius: 10%;
    box-shadow: 2px 2px 5px #3066be;
}

/* Vibrant headers */
h2, h3, h4 {
    color: #fff;
}

/* Custom styled file uploader */
.stFileUploader > div > label {
    background-color: #FF4500;
    color: white;
    font-size: 16px;
    padding: 10px 15px;
    border-radius: 10px;
    cursor: pointer;
    transition: transform .2s;
}

.stFileUploader > div > label:hover {
    transform: scale(1.1);
}

h1 .css-10trblm {
    background: -webkit-linear-gradient(#090c9b, #3066be);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

st.title('Fashion Outfit Generator')

st.subheader("Upload Your Fashion Item 📸")

progress_bar = st.progress(0)
for percent_complete in range(100):
    progress_bar.progress(percent_complete + 1)


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


st.success("Choose an image")
uploaded_file = st.file_uploader('')


def my_widget():
    st.subheader('Hello there!')


if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):

        st.subheader("Uploaded Fashion Item:")
        display_image = Image.open(uploaded_file)
        st.image(display_image, width=250)

        features = feature_extraction(os.path.join(
            "uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)

        st.subheader("Fashion Recommendations:")

        cols = st.columns(5)

        for idx, col in enumerate(cols):
            with col:
                with st.container():
                    rec_image = Image.open(filenames[indices[0][idx]])
                    st.image(rec_image, use_column_width=True,
                             caption=f"Recommendation #{idx + 1}")

    else:
        st.header("Some error occurred in file upload")
