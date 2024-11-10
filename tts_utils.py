from elevenlabs.client import ElevenLabs
import streamlit as st
import os
from dotenv import load_dotenv
import base64

load_dotenv()

def initialize_session_state():
    """Initialize all session state variables"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'solution_json' not in st.session_state:
        st.session_state.solution_json = ""
    if 'step_by_step_solution' not in st.session_state:
        st.session_state.step_by_step_solution = ""
    if 'processing_image' not in st.session_state:
        st.session_state.processing_image = False
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = False
    if 'solving_problem' not in st.session_state:
        st.session_state.solving_problem = False
    if 'audio_key' not in st.session_state:
        st.session_state.audio_key = 0

def stream_audio_response(text):
    """
    Generate audio response using ElevenLabs API
    Returns the base64 encoded audio data
    """
    try:
        if not text:
            return None
            
        client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
        
        audio_stream = client.generate(
            text=text,
            voice="Nicole",
            model="eleven_turbo_v2_5",
            stream=True
        )
        
        audio_data = b''
        for chunk in audio_stream:
            audio_data += chunk
        
        if not audio_data:
            return None
            
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        if audio_base64:
            # Play audio with controls instead of autoplay
            st.markdown(
                f'''<div class="audio-player">
                    <audio controls style="width: 100%;">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                </div>''',
                unsafe_allow_html=True
            )
            
            return audio_base64
    
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None