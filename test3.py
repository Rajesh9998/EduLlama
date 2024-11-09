import streamlit as st
from streamlit_mic_recorder import mic_recorder
from groq import Groq
from tempfile import NamedTemporaryFile
import os
from dotenv import load_dotenv

load_dotenv()
# Set page title
st.title("Audio Transcription with Groq")

# Initialize Groq client - make sure to set your GROQ_API_KEY environment variable
client = Groq(
    api_key=os.getenv('GROQ_API_KEY')
)

def transcribe_audio(audio_bytes, filename):
    """Function to handle Groq transcription"""
    try:
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
        return transcription.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

# Initialize session state for storing transcriptions
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []

# Create the audio recorder
audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    key="recorder"
)

# Process the audio when recorded
if audio:
    with st.spinner("Processing audio..."):
        try:
            # Create a temporary file for the audio
            with NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                # Write the audio bytes to the file
                tmp_file.write(audio['bytes'])
                tmp_file.flush()
                
                # Display the recorded audio
                st.subheader("Recorded Audio:")
                st.audio(audio['bytes'])
                
                # Transcribe using Groq
                transcription = transcribe_audio(audio['bytes'], tmp_file.name)
                
                if transcription:
                    # Add to transcription history
                    st.session_state.transcriptions.append(transcription)
                    
                    # Display the latest transcription
                    st.subheader("Latest Transcription:")
                    st.write(transcription)
                    
                    # Display transcription history
                    if len(st.session_state.transcriptions) > 1:
                        st.subheader("Transcription History:")
                        for i, text in enumerate(st.session_state.transcriptions[:-1], 1):
                            st.text(f"Recording {i}: {text}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Clean up the temporary file
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)

# Add instructions
st.markdown("""
### Instructions:
1. Click the '🎤 Start Recording' button to begin
2. Speak clearly into your microphone
3. Click '⏹️ Stop Recording' when finished
4. Wait for Groq to process and transcribe your audio
5. View your transcription below the audio player

Note: Make sure you have:
- Allowed microphone access in your browser
- Set up your GROQ_API_KEY environment variable
""")