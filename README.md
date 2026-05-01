# CIFAR-10 Image Classification: Custom CNN vs. VGG16 Transfer Learning
## An interactive web application comparing a custom-built CNN and VGG16 Transfer Learning for multi-class image classification on the CIFAR-10 dataset.[cite: 1]

### Project Overview
This personal project focuses on solving the multi-class image classification problem using the **CIFAR-10 dataset**. The objective was to compare two distinct deep learning strategies:
1.  **Custom CNN**: A "from-scratch" architecture designed with three convolutional blocks, batch normalization, and dropout to handle small-scale image data.
2.  **Transfer Learning (VGG16)**: Leveraging a pre-trained VGG16 model (ImageNet weights) to utilize high-level feature extractions for improved accuracy.

---
### Repository Structure
```text
CIFAR10-Image-Classification-CNN-vs-VGG16/
├── notebook/
│   └── Image_Classification_CNN_vs_VGG16_CIFAR10.ipynb  # Training and evaluation logic
├── app.py                                              # Streamlit application script
├── cnn_cifar10_model.keras                             # Saved Custom CNN model weights
├── vgg16_cifar10_model.keras                           # Saved VGG16 Transfer Learning model weights
├── requirements.txt                                    # List of dependencies for deployment
├── README.md                                           # Project documentation and live demo link
├── .gitattributes                                      # Git LFS configuration for large model files
└── .gitignore                                          # Specifies files for Git to ignore (e.g., venv, __pycache__)
```

---
### Insights & Observations
Based on the experimental results:
*   **Performance Gap:** The **VGG16 model (~82.4% accuracy)** significantly outperformed the **Custom CNN (~76% accuracy)**. This highlights the power of pre-trained spatial hierarchies in transfer learning.
*   **Training Stability:** The VGG16 model showed more stable validation loss curves, whereas the Custom CNN exhibited higher fluctuations, indicating it was more sensitive to hyperparameter changes.
*   **Data Augmentation:** Implementing `ImageDataGenerator` for rotation and flipping was critical in reducing overfitting for the Custom CNN, allowing it to reach a respectable accuracy despite having fewer parameters than VGG16.

---
### Recommendations
*   **Architecture for Production:** For real-world applications with limited data, **Transfer Learning** is the clear recommendation as it yields higher accuracy with less training time.
*   **Future Scope:** Further improvements could be made by testing even deeper architectures like **ResNet50** or **EfficientNet** to see if the small 32x32 image size benefits from residual connections.

---
**An interactive web application comparing a custom-built CNN and VGG16 Transfer Learning for multi-class image classification on the CIFAR-10 dataset.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cifar10-image-classification-cnn-vs-vgg16-wqpwgzwbhfchww2eu2i8.streamlit.app/)

---
## 🛠️ Installation & Setup
To run the Streamlit app locally:
1.  Clone the repository:
    ```bash
    git clone [https://github.com/digvijay2420/CIFAR10-Image-Classification-CNN-vs-VGG16.git](https://github.com/digvijay2420/CIFAR10-Image-Classification-CNN-vs-VGG16.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Launch the app:
    ```bash
    streamlit run app.py
    ```

---
### Academic Information
- **Name:** Digvijay Singh
- **Affiliations:**
  - Diploma in Data Science, IIT Madras (IITM)
  - Business Analytics Fellow, HEProAI
- **Libraries & Tools Used:**
  - Frameworks: TensorFlow, Keras  
  - Data Manipulation: NumPy, Pandas  
  - Visualization: Matplotlib, Seaborn  
  - Deployment: Streamlit Cloud  
  - Image Processing: Pillow (PIL)  
  - Environments: Google Colab (T4 GPU), VS Code
