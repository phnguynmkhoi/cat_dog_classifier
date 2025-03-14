import streamlit as st
import requests
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐾",
    layout="centered"
)

# Define API URL
API_URL = "http://localhost:8000/predict/"

def main():
    st.title("Dog vs Cat Classifier")
    st.write("Upload an image and our AI will tell you if it's a dog or a cat!")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Add a predict button
        if col1.button("Predict"):
            with st.spinner("Analyzing image..."):
                
                try:
                    uploaded_file.seek(0) 
                    file_bytes = uploaded_file.getvalue()

                    # Send file as multipart/form-data
                    files = {
                        "file": (uploaded_file.name, file_bytes, uploaded_file.type)  # Correct format
                    }

                    # Send request to FastAPI
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["prediction"]
                        confidence = result["confidence"]
                        
                        # Display prediction result
                        col2.subheader("Prediction Results")
                        
                        # Display the prediction text
                        col2.markdown(f"<h3 style='text-align: center;'>It's a {prediction.upper()}!</h3>", unsafe_allow_html=True)
                        col2.markdown(f"<p style='text-align: center;'>Confidence: {confidence}%</p>", unsafe_allow_html=True)
                        
                    else:
                        col2.error(f"Error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    col2.error(f"Error connecting to API: {str(e)}")
                    col2.info("Make sure your FastAPI backend is running at " + API_URL)
    
    # Add information section
    with st.expander("About this app"):
        st.write("""
        This app uses a deep learning model to classify images as either dogs or cats.
        
        The model is a fine-tuned ResNet18 neural network that has been trained on thousands of dog and cat images.
        
        To use the app:
        1. Upload an image using the file uploader
        2. Click the 'Predict' button
        3. View the results!
        """)

if __name__ == "__main__":
    main()