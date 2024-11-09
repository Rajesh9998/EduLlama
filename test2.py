import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import io
from test import solve_math_problem, extract_question_from_image

load_dotenv()

# Set page config and dark theme
st.set_page_config(page_title="Math Problem Solver", layout="wide")

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

def main():
    st.title("Math Problem Solver")
    
    # Create tabs for Image and Text
    tab1, tab2 = st.tabs(["📸 Image Input", "✍️ Text Input"])
    
    with tab1:
        st.subheader("Upload Image with Math Problem")
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Solve Problem from Image"):
                with st.spinner("Processing image and solving problem..."):
                    # Save image temporarily
                    temp_path = "temp_image.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Extract question from image
                        question_text = extract_question_from_image(temp_path)
                        st.write("Extracted Question:")
                        st.write(question_text)
                        
                        # Solve the extracted problem
                        solution = solve_math_problem(question_text)
                        
                        st.write("Solution:")
                        st.write(solution)
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    with tab2:
        st.subheader("Enter Math Problem")
        user_text = st.text_area("Enter your math problem here", height=200)
        
        if st.button("Solve Problem"):
            with st.spinner("Solving problem..."):
                solution = solve_math_problem(user_text)
                st.write("Solution:")
                st.write(solution)

if __name__ == "__main__":
    main()