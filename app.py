import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="wide")
st.title("CIFAR-10 Image Classification: CNN vs. VGG16")

# 1. Load Models (Modern .keras format)
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model('cnn_cifar10_model.keras')
    vgg = tf.keras.models.load_model('vgg16_cifar10_model.keras')
    return cnn, vgg
try:
    cnn_model, vgg_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure .keras files are in the repository.")
# 2. Define CIFAR-10 Classes
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 3. Sidebar for Additional Information
st.sidebar.header("About the Project")
st.sidebar.info("""
- **Dataset:** CIFAR-10 (32x32 RGB images)
- **Framework:** TensorFlow / Keras
- **Deployment:** Streamlit Cloud
""")

# 4. Image Upload Interface
uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, or PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create two columns for a clean layout
    main_col1, main_col2 = st.columns([1, 2])
    
    with main_col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    # 5. Preprocessing (matching training steps)
    # Resize to 32x32 and normalize pixel values
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 6. Classification Action
    if st.button('🔍 Classify Image'):
        with st.spinner('Calculating predictions...'):
            # Perform Predictions
            pred_cnn = cnn_model.predict(img_array)
            pred_vgg = vgg_model.predict(img_array)
            
            # Display Results in columns
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("### 🏗️ Custom CNN")
                label_cnn = class_names[np.argmax(pred_cnn)]
                confidence_cnn = np.max(pred_cnn) * 100
                st.metric(label="Predicted Class", value=label_cnn)
                st.progress(float(confidence_cnn / 100))
                st.write(f"Confidence: {confidence_cnn:.2f}%")

            with res_col2:
                st.markdown("### 🎓 VGG16 Transfer Learning")
                label_vgg = class_names[np.argmax(pred_vgg)]
                confidence_vgg = np.max(pred_vgg) * 100
                st.metric(label="Predicted Class", value=label_vgg)
                st.progress(float(confidence_vgg / 100))
                st.write(f"Confidence: {confidence_vgg:.2f}%")