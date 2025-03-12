import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow model predictiion
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Choose the page", ["Home", "About", "Disease Recognition"], index=0)

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this [page](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download)!.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. The model is trained on 70295 images as of now.
                2. Testing is done on 33 images.
                3. Model is validated on 17572 images.
                #### 

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Strawberry___healthy', 'coffee_rust_level_3', 'chilli_healthy', 'Cassava_mosaic', 'guava_rust', 'Grape___Black_rot', 'Potato___Early_blight', 'basil_healthy', 'Blueberry___healthy', 'coffee_rust_level_2', 'guava_canker', 'Corn_(maize)___healthy', 'tea_red_leaf_spot', 'Tomato___Target_Spot', 'cotton_healthy', 'chilli_leaf_curl', 'basil_wilted', 'guava_mummification', 'cassava_mosaic_disease', 'Peach___healthy', 'Maize_fall_armyworm', 'Potato___Late_blight', 'tea_red_scab', 'cotton_bacterial_blight', 'Tomato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Pepper,_bell___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Tomato___Leaf_Mold', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'chilli_leaf_spot', 'wheat_healthy', 'Cassava_green_mite', 'Cherry_(including_sour)___Powdery_mildew', 'banana_xamthomonas', 'Apple___Cedar_apple_rust', 'citrus__black_spot', 'lettuce_bacterial_spot', 'Tomato___Bacterial_spot', 'tea_leaf_blight', 'Grape___healthy', 'citrus_canker', 'banana_segatoka', 'coffee_red_spider_mite', 'Cashew_leaf_miner', 'Maize_grasshoper', 'guava_dot', 'chilli_yellowish', 'citrus_greening', 'lettuce_anthracnose', 'cassava_brown_streak_disease', 'parsley_leaf_blight_disease', 'Tomato___Early_blight', 'parsley_leaf_spot_disease', 'citrus_healthy', 'Corn_(maize)___Common_rust_', 'mint_leaf_rust', 'cotton_curl_virus', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'coffee_rust_level_1', 'lettuce_soft_rot', 'basil_with_mildew', 'bean_angular_leaf_spot', 'Tomato___healthy', 'coffee_healthy', 'cassava_healthy', 'Cherry_(including_sour)___healthy', 'brassica_black_rot', 'rice_leaf_smut', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Maize_streak_virus', 'lettuce_downy_mildew', 'cassava_green_mottle', 'rice_bacterial_leaf_blight', 'cotton_fussarium_wilt', 'Apple___Apple_scab', 'Cassava_bacterial_blight', 'mint_fusarium_wilt', 'Corn_(maize)___Northern_Leaf_Blight', 'Maize_leaf_beetle', 'powdery_mildew_mint_leaf', 'Cashew_red_rust', 'banana_healthy', 'Tomato___Spider_mites Two-spotted_spider_mite', 'rice_brown_spot', 'Cassava_brown_spot', 'wheat_septoria', 'Peach___Bacterial_spot', 'bean_healthy', 'Cashew_gumosis', 'Pepper,_bell___Bacterial_spot', 'guava_healthy', 'kale_with_spots', 'Tomato___Septoria_leaf_spot', 'citrus_melanose', 'wheat_stripe_rust', 'coriander_healthy', 'Squash___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Cashew_anthracnose', 'Apple___Black_rot', 'bean_rust', 'Apple___healthy', 'chilli_whitefly', 'Cashew_healthy', 'Strawberry___Leaf_scorch', 'Potato___healthy', 'Soybean___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))