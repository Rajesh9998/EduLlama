import streamlit as st
st.set_page_config(page_title="Math Problem Solver with Voice Interaction", layout="wide")
import os
from PIL import Image
from dotenv import load_dotenv
from test import solve_math_problem, extract_question_from_image, run_together_llm
from test3 import transcribe_audio
from streamlit_mic_recorder import mic_recorder
from groq import Groq

load_dotenv()

# Initialize Groq client
client = Groq()
genai_api_key = os.environ.get("GEMINI_API_KEY")

def main():
    st.title("Math Problem Solver")

    # Initialize session state for conversation and problem data
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'solution_json' not in st.session_state:
        st.session_state.solution_json = ""
    if 'step_by_step_solution' not in st.session_state:
        st.session_state.step_by_step_solution = ""

    # Create two columns
    col1, col2 = st.columns(2)

    with col2:
        st.subheader("Problem Input")
        # Create tabs for Image and Text
        tab1, tab2 = st.tabs(["📸 Image Input", "✍️ Text Input"])

        with tab1:
            uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                if st.button("Solve Problem from Image"):
                    with st.spinner("Processing image and solving problem..."):
                        temp_path = "temp_image.jpg"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        try:
                            # Extract question from image
                            question_text = extract_question_from_image(temp_path)
                            st.write("**Extracted Question:**")
                            st.write(question_text)

                            # Solve the extracted problem
                            solution_json = solve_math_problem(question_text)

                            # Generate step-by-step solution using together model
                            step_by_step_solution = run_together_llm(
                                "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                                f"Please provide a step-by-step solution in markdown format for the following problem:\n{question_text}\nSolution JSON:\n{solution_json}"
                            )

                            # Display the solution
                            st.markdown("### Step-by-Step Solution:")
                            st.markdown(step_by_step_solution)

                            # Save data in session state
                            st.session_state.question = question_text
                            st.session_state.solution_json = solution_json
                            st.session_state.step_by_step_solution = step_by_step_solution
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

        with tab2:
            st.subheader("Enter Math Problem")
            user_text = st.text_area("Enter your math problem here", height=200)
            if st.button("Solve Problem"):
                with st.spinner("Solving problem..."):
                    question_text = user_text
                    solution_json = solve_math_problem(user_text)

                    # Generate step-by-step solution using together model
                    step_by_step_solution = run_together_llm(
                        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                        f"Please provide a step-by-step solution in markdown format for the following problem:\n{question_text}\nSolution JSON:\n{solution_json}"
                    )

                    # Display the solution
                    st.markdown("### Step-by-Step Solution:")
                    st.markdown(step_by_step_solution)

                    # Save data in session state
                    st.session_state.question = question_text
                    st.session_state.solution_json = solution_json
                    st.session_state.step_by_step_solution = step_by_step_solution

    with col1:
        st.subheader("Voice Interaction")
        # Voice recording
        audio = mic_recorder(
            start_prompt="🎤 Start Speaking",
            stop_prompt="⏹️ Stop Speaking",
            just_once=False,
            key="voice_recorder"
        )
        if audio:
            with st.spinner("Processing audio..."):
                try:
                    # Transcribe audio
                    transcribed_query = transcribe_audio(audio['bytes'], "audio.wav")

                    # Display user's query
                    st.markdown(f"**User:** {transcribed_query}")

                    # Append to conversation history
                    st.session_state.conversation.append(
                        {"role": "user", "content": transcribed_query}
                    )

                    # Prepare messages for Groq AI
                    messages = st.session_state.conversation.copy()
                    # Add context to the beginning
                    context = f"The original math problem is:\n{st.session_state.question}\n\nThe step-by-step solution is:\n{st.session_state.step_by_step_solution}\n"
                    messages.insert(0, {"role": "system", "content": context})

                    # Generate assistant's response using Groq AI
                    completion = client.chat.completions.create(
                        model="llama-3.2-90b-text-preview",
                        messages=messages,
                        temperature=1,
                        max_tokens=1024,
                        top_p=1,
                        stream=False,
                        stop=None,
                    )
                    assistant_response = completion.choices[0].message.content

                    # Display assistant's response
                    st.markdown(f"**Assistant:** {assistant_response}")

                    # Append assistant's response to conversation
                    st.session_state.conversation.append(
                        {"role": "assistant", "content": assistant_response}
                    )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()