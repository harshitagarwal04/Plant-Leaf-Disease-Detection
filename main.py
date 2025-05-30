import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("🌱 Plant Disease Dashboard")
    st.markdown("---")
    app_mode = st.selectbox("📄 Choose the page", ["Home", "About", "Disease Recognition"], index=0)
    st.markdown("---")
    st.info("Developed by Team GreenAI 🌿")

# Load model once and cache it
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 97)  # 91 classes
    model.load_state_dict(torch.load("resnet50_plant_disease.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image preprocessing for ResNet50
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image).convert('RGB')
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# PyTorch model prediction
def model_prediction(test_image):
    input_tensor = preprocess_image(test_image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

# Class names
class_name = [
    "Apple Blackrot", "Apple Cedar Apple Rust", "Apple Healthy", "Apple Scab", "Banana Healthy", "Banana Segatoka",
    "Banana Xamthomonas", "Basil Wilted", "Basil With Mildew", "Bean Angular Leaf Spot", "Bean Healthy", "Bean Rust",
    "Blueberry Healthy", "Brassica Black Rot", "Cassava Bacterial Blight", "Cassava Brown Streak Disease",
    "Cassava Green Mottle", "Cassava Healthy", "Cassava Mosaic Disease", "Cherry Healthy", "Cherry Powdery Mildew",
    "Chilli Healthy", "Chilli Leaf Curl", "Chilli Leaf Spot", "Chilli Whitefly", "Chilli Yellowish", "Citrus Black Spot",
    "Citrus Canker", "Citrus Greening", "Citrus Healthy", "Citrus Melanose", "Coffee Healthy", "Coffee Red Spider Mite",
    "Coffee Rust Level 1", "Coffee Rust Level 2", "Coffee Rust Level 3", "Corn Cercospora Leaf Spot", "Corn Common Rust",
    "Corn Healthy", "Corn Northern Leaf Blight", "Cotton Bacterial Blight", "Cotton Curl Virus", "Cotton Fussarium Wilt",
    "Cotton Healthy", "Grape Black Rot", "Grape Esca Black Measles", "Grape Healthy", "Grape Leaf Blight Isariopsis Leaf Spot",
    "Guava Canker", "Guava Dot", "Guava Healthy", "Guava Mummification", "Guava Rust", "Healthy Basil", "Healthy Coriander",
    "Kale With Spots", "Lettuce Anthracnose", "Lettuce Bacterial Spot", "Lettuce Downy Mildew", "Lettuce Soft Rot",
    "Mint Fusarium Wilt", "Mint Leaf Rust", "Orange Haunglongbing Citrus Greening", "Parsley Leaf Blight Disease",
    "Parsley Leaf Spot Disease", "Peach Bacterial Spot", "Peach Healthy", "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
    "Potato Early Blight", "Potato Healthy", "Potato Late Blight", "Powdery Mildew Mint Leaf", "Raspberry Healthy",
    "Rice Bacterial Leaf Blight", "Rice Brown Spot", "Rice Leaf Smut", "Soybean Healthy", "Squash Powdery Mildew",
    "Strawberry Healthy", "Strawberry Leaf Scorch", "Tea Leaf Blight", "Tea Red Leaf Spot", "Tea Red Scab",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight", "Tomato Leaf Mold",
    "Tomato Mosaic Virus", "Tomato Septoria Leaf Spot", "Tomato Spider Mites Two Spotted Spider Mite", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Wheat Healthy", "Wheat Septoria", "Wheat Stripe Rust"
]

# Home Page
if app_mode == "Home":
    st.markdown("<h1 style='color:#228B22;'>🌿 Plant Disease Recognition System</h1>", unsafe_allow_html=True)
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
    <div style='font-size:18px;'>
    Welcome to the <b>Plant Disease Recognition System</b>!<br>
    <ul>
        <li>🌱 <b>Upload</b> a plant image on the <b>Disease Recognition</b> page.</li>
        <li>🔬 <b>Analyze</b> plant health using AI-powered detection.</li>
        <li>📊 <b>Get instant results</b> and recommendations.</li>
    </ul>
    <b>Why Choose Us?</b>
    <ul>
        <li>✅ <b>Accurate</b> deep learning model</li>
        <li>⚡ <b>Fast</b> and easy to use</li>
        <li>🖼️ <b>Modern</b> interface</li>
    </ul>
    <b>Get Started:</b> Go to <b>Disease Recognition</b> in the sidebar!
    </div>
    """, unsafe_allow_html=True)

# About Page
elif app_mode == "About":
    st.markdown("<h1 style='color:#228B22;'>About</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:17px;'>
    <b>Dataset:</b><br>
    - 88,000+ RGB images of healthy and diseased crop leaves<br>
    - 97 different classes<br>
    - 80/20 train-validation split<br>
    - 33 test images for prediction<br>
    <br>
    <b>Model:</b><br>
    - Trained on 88,327 images<br>
    - Validated on 25,682 images<br>
    - Based on <b>ResNet50</b> deep learning architecture<br>
    <br>
    <b>Source:</b> <a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download" target="_blank">Kaggle Plant Disease Dataset</a>
    </div>
    """, unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown("<h1 style='color:#228B22;'>🦠 Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("Browse all possible disease classes, upload a clear image of a plant leaf, and detect possible diseases.")

    # 1. Browse all classes (on top)
    st.markdown("#### 📚 Browse All Classes")
    st.selectbox("All possible disease classes", class_name, index=0, key="class_select")

    # 2. Upload and preview section (side by side)
    col1, col2 = st.columns([2, 1])
    with col1:
        test_image = st.file_uploader("📷 Upload a Plant Leaf Image", type=["jpg", "jpeg", "png"])
        predict_btn = st.button("🔍 Predict Disease", use_container_width=True)
    with col2:
        if test_image:
            st.markdown("##### Image Preview")
            st.image(test_image, caption="Uploaded Image", use_container_width=False, width=180)

    # 3. Prediction section
    if test_image and predict_btn:
        with st.spinner("Analyzing image..."):
            pred_idx, confidence = model_prediction(test_image)
        st.snow()
        st.success(f"🌱 **Prediction:** {class_name[pred_idx]}")
        st.progress(int(confidence * 100))
        st.info(f"Model confidence: **{confidence*100:.2f}%**")
    elif not test_image and predict_btn:
        st.warning("Please upload a plant leaf image to analyze.")

# Footer
st.markdown("""
    <hr>
    <center>
    <span style='font-size:15px;'>Made with ❤️ by Team GreenAI | 2025</span>
    </center>
    """, unsafe_allow_html=True)