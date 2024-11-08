import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import io

load_dotenv()

# Set page config and dark theme
st.set_page_config(page_title="AI Assistant - Image & Text Analysis", layout="wide")

# Custom CSS for dark theme and tabs
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSidebar {
        background-color: #2D2D2D;
    }
    .stTabs {
        background-color: #2D2D2D;
        border-radius: 4px;
    }
    .stTextInput {
        background-color: #2D2D2D;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def analyze_image(image_path):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",
        generation_config=generation_config,
    )

    file = genai.upload_file(image_path)

    chat = model.start_chat()
    response = chat.send_message([file, "extract the question and options as it is from the image. Note: Don't return any text other than the Question and options."])
    return response.text

def analyze_text(text):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",
        generation_config=generation_config,
    )

    chat = model.start_chat()
    response = chat.send_message(text)
    return response.text

def main():
    st.title("AI Assistant - Image & Text Analysis")
    
    # Create tabs for Image and Text
    tab1, tab2 = st.tabs(["📸 Image Analysis", "✍️ Text Analysis"])
    
    with tab1:
        st.subheader("Upload or Paste Image for Analysis")
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

        # To allow pasting an image from the clipboard (via file upload)
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    temp_path = "temp_image.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    result = analyze_image(temp_path)
                    st.write("Analysis Results:")
                    st.write(result)
                    
                    os.remove(temp_path)

        else:
            st.warning("Please upload an image file for analysis.")

    with tab2:
        st.subheader("Text Analysis - Enter or Paste Text Below")
        user_text = st.text_area("Enter your text here for analysis", height=200)

        if st.button("Analyze Text"):
            with st.spinner("Analyzing text..."):
                result = analyze_text(user_text)
                st.write("Analysis Results:")
                st.write(result)

if __name__ == "__main__":
    main()


