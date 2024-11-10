import streamlit as st
st.set_page_config(page_title="Math Problem Solver with Voice Interaction", layout="wide")
import os
from PIL import Image
from dotenv import load_dotenv
from test import solve_math_problem, extract_question_from_image, run_together_llm
from test3 import transcribe_audio
from streamlit_mic_recorder import mic_recorder
from groq import Groq
from tts_utils import generate_audio_response
import elevenlabs

load_dotenv()

# Initialize Groq client
client = Groq()
# Initialize ElevenLabs
elevenlabs.ElevenLabs(api_key= os.getenv('ELEVENLABS_API_KEY'))
# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'solution_json' not in st.session_state:
    st.session_state.solution_json = ""
if 'step_by_step_solution' not in st.session_state:
    st.session_state.step_by_step_solution = ""
if 'audio_counter' not in st.session_state:
    st.session_state.audio_counter = 0

def main():
    st.title("Math Problem Solver")

    # Create two columns (Problem Input on left, Voice Interaction on right)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Problem Input")
        # Create tabs for Image and Text
        tab1, tab2 = st.tabs(["📸 Image Input", "✍️ Text Input"])

        with tab1:
            uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                if st.button("Solve Problem from Image"):
                    with st.spinner("Processing image and solving problem..."):
                        temp_path = "temp_image.jpg"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        try:
                            question_text = extract_question_from_image(temp_path)
                            st.write("**Extracted Question:**")
                            st.write(question_text)

                            solution_json = solve_math_problem(question_text)
                            step_by_step_solution = run_together_llm(
                                "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                                f"Please provide a step-by-step solution in markdown format for the following problem:\n{question_text}\nSolution JSON:\n{solution_json}"
                            )

                            st.session_state.question = question_text
                            st.session_state.solution_json = solution_json
                            st.session_state.step_by_step_solution = step_by_step_solution
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

        with tab2:
            user_text = st.text_area("Enter your math problem here", height=200)
            if st.button("Solve Problem"):
                with st.spinner("Solving problem..."):
                    question_text = user_text
                    solution_json = solve_math_problem(user_text)
                    step_by_step_solution = run_together_llm(
                        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                        f"Please provide a step-by-step solution in markdown format for the following problem:\n{question_text}\nSolution JSON:\n{solution_json}"
                    )

                    st.session_state.question = question_text
                    st.session_state.solution_json = solution_json
                    st.session_state.step_by_step_solution = step_by_step_solution

        # Display the step-by-step solution if available
        if st.session_state.step_by_step_solution:
            st.markdown("### Step-by-Step Solution:")
            st.markdown(st.session_state.step_by_step_solution)

    with col2:
        st.subheader("Voice Interaction")
        # Conversation container to display messages
        conversation_container = st.container()

        # Voice recording container at the bottom
        with st.container():
            audio = mic_recorder(
                start_prompt="🎤 Start Speaking",
                stop_prompt="⏹️ Stop Speaking",
                just_once=True,
                key="voice_recorder"
            )

            if audio:
                with st.spinner("Processing audio..."):
                    try:
                        transcribed_query = transcribe_audio(audio['bytes'], "audio.wav")
                        st.session_state.conversation.append(
                            {"role": "user", "content": transcribed_query}
                        )

                        messages = st.session_state.conversation.copy()
                        context = f"The original math problem is:\n{st.session_state.question}\n\nThe step-by-step solution is:\n{st.session_state.step_by_step_solution}\n"
                        messages.insert(0, {"role": "system", "content": context})

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

                        st.session_state.conversation.append(
                            {"role": "assistant", "content": assistant_response}
                        )

                        # Generate audio response
                        audio_response = generate_audio_response(assistant_response)
                        if audio_response:
                            st.session_state.audio_counter += 1
                            st.audio(audio_response, format='audio/mp3')
                            # Add autoplay script
                            st.markdown(
                                f"""
                                <script>
                                    const audioElements = document.querySelectorAll('audio');
                                    const lastAudio = audioElements[audioElements.length - 1];
                                    if (lastAudio) {{
                                        lastAudio.play();
                                    }}
                                </script>
                                """,
                                unsafe_allow_html=True
                            )

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

        # Display conversation history after processing
        with conversation_container:
            for message in st.session_state.conversation:
                st.markdown(f"**{message['role'].title()}:** {message['content']}")

if __name__ == "__main__":
    main()