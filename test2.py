import streamlit as st
# This must be the first Streamlit command
st.set_page_config(
    page_title="Math Problem Solver with Voice Interaction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import os
from PIL import Image
from dotenv import load_dotenv
from test import solve_math_problem, extract_question_from_image, run_together_llm
from test3 import transcribe_audio
from groq import Groq
from tts_utils import stream_audio_response, initialize_session_state
import elevenlabs

# Load environment variables
load_dotenv()

# Initialize clients
client = Groq()
elevenlabs.ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

# Update custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
        color: #1a1a1a;
    }
    .question-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #d1d1d1;
        color: #1a1a1a;
    }
    .solution-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #b8daff;
        color: #1a1a1a;
    }
    .chat-message {
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #d1d1d1;
        color: #1a1a1a;
    }
    .user-message {
        background-color: #ffffff;
        margin-right: 2rem;
    }
    .assistant-message {
        background-color: #ffffff;
        margin-left: 2rem;
    }
    .conversation-area {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #d1d1d1;
        margin-bottom: 1rem;
        max-height: 500px;
        overflow-y: auto;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

def process_image(uploaded_file):
    """Process the uploaded image and extract text"""
    temp_path = "temp_image.jpg"
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        question_text = extract_question_from_image(temp_path)
        st.session_state.question = question_text
        st.session_state.image_processed = True
        return question_text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_problem(question_text):
    """Process the math problem and generate solution"""
    solution_json = solve_math_problem(question_text)
    step_by_step_solution = run_together_llm(
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        f"Please provide a step-by-step solution in markdown format for:\n{question_text}\nSolution JSON:\n{solution_json}"
    )
    st.session_state.question = question_text
    st.session_state.solution_json = solution_json
    st.session_state.step_by_step_solution = step_by_step_solution
    st.session_state.solving_problem = False

def display_conversation():
    """Display the conversation history with audio players"""
    for idx, message in enumerate(st.session_state.conversation):
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">🗣️ <b>You:</b> {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message">🤖 <b>Assistant:</b> {message["content"]}</div>',
                unsafe_allow_html=True
            )
            
            # Display associated audio if it exists
            if "audio_base64" in message:
                st.markdown(
                    f"""
                    <div class="audio-player">
                        <audio controls style="width: 100%;">
                            <source src="data:audio/mp3;base64,{message['audio_base64']}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def process_voice_input(audio):
    """Process voice input and generate response"""
    try:
        transcribed_query = transcribe_audio(audio['bytes'], "audio.wav")
        
        # Add user message to display
        st.session_state.conversation.append(
            {"role": "user", "content": transcribed_query}
        )

        # Prepare messages for API without audio_base64
        api_messages = [
            {"role": "system", "content": f"Question: {st.session_state.question}\nSolution: {st.session_state.step_by_step_solution}"}
        ]
        
        # Add conversation history without audio_base64
        for msg in st.session_state.conversation:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        completion = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=api_messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        assistant_response = completion.choices[0].message.content
        
        # Generate and store audio response
        audio_base64 = stream_audio_response(assistant_response)
        
        # Add assistant message with audio to display
        st.session_state.conversation.append({
            "role": "assistant", 
            "content": assistant_response,
            "audio_base64": audio_base64
        })
        
        st.rerun()

    except Exception as e:
        st.error(f"Error processing voice input: {str(e)}")

def main():
    st.title("🧮 Interactive Math Problem Solver")

    # Initialize session state
    initialize_session_state()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Problem Input")
        tabs = st.tabs(["📸 Image Input", "✍️ Text Input"])
        
        with tabs[0]:
            uploaded_file = st.file_uploader("Upload an image of your math problem", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                if not st.session_state.image_processed:
                    question_text = process_image(uploaded_file)
                
                if st.session_state.image_processed:
                    st.success("Text extracted successfully!")
                    st.markdown("### 📝 Extracted Question")
                    st.markdown(f'<div class="question-box">{st.session_state.question}</div>', unsafe_allow_html=True)
                    
                    if not st.session_state.solving_problem and st.button("🧮 Solve Problem", key="solve_image"):
                        st.session_state.solving_problem = True
                        process_problem(st.session_state.question)

        with tabs[1]:
            user_text = st.text_area("Type your math problem here", height=150)
            if st.button("🧮 Solve Problem", key="solve_text") and user_text:
                process_problem(user_text)

        # Ensure solution is always visible when available
        if st.session_state.step_by_step_solution:
            st.markdown("### 🔍 Solution")
            solution_html = f'''
                <div class="solution-box">
                    <div style="color: #1a1a1a;">
                        {st.session_state.step_by_step_solution}
                    </div>
                </div>
            '''
            st.markdown(solution_html, unsafe_allow_html=True)

    with col2:
        st.header("🎙️ Voice Assistant")
        from streamlit_mic_recorder import mic_recorder
        
        # Conversation area
        with st.container():
            st.markdown('<div class="conversation-area">', unsafe_allow_html=True)
            display_conversation()
            st.markdown('</div>', unsafe_allow_html=True)

        # Voice controls
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                audio = mic_recorder(
                    start_prompt="Start Speaking",
                    stop_prompt="Stop Recording",
                    just_once=True,
                    key=f"voice_recorder_{st.session_state.audio_key}"
                )
            with col2:
                if st.button("Clear Chat"):
                    st.session_state.conversation = []
                    st.session_state.audio_key = 0
                    st.rerun()
        
        if audio:
            process_voice_input(audio)
            st.session_state.audio_key += 1

if __name__ == "__main__":
    main()