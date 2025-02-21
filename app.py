import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

def cartoonize_image(
    img_array, 
    n_colors=8, 
    max_iter=20, 
    epsilon=0.001, 
    attempts=10, 
    blur_kernel_size=7, 
    block_size=9, 
    C=2
):
    # Convert to OpenCV format
    img = cv2.cvtColor(np.array(img_array), cv2.COLOR_RGB2BGR)
    
    # Color quantization using K-Means
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    _, labels, centers = cv2.kmeans(
        data, n_colors, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
    )
    quantized = np.uint8(centers)[labels.flatten()].reshape(img.shape)
    
    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blur_kernel_size)
    edges = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
    )
    
    # Combine quantized colors with edges
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
    cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    return cartoon_rgb

# Streamlit app interface
st.title("Cartoonizer App 🎨")
st.write("Upload an image, tweak the parameters, and see the magic happen!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)
    
    st.sidebar.header("Cartoonization Parameters")
    n_colors = st.sidebar.slider("Number of Colors", 2, 20, 8)
    blur_kernel_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 7, step=2)
    block_size = st.sidebar.slider("Block Size for Thresholding", 3, 15, 9, step=2)
    C = st.sidebar.slider("Edge Detection Constant (C)", 0, 10, 2)

    if st.button("Cartoonize!"):
        cartoon = cartoonize_image(
            img, 
            n_colors=n_colors, 
            blur_kernel_size=blur_kernel_size, 
            block_size=block_size, 
            C=C
        )
        st.image(cartoon, caption="Cartoonized Image", use_column_width=True)
