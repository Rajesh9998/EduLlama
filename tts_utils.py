import elevenlabs
from io import BytesIO


def generate_audio_response(text):
    """Generate audio from text using ElevenLabs"""
    


    try:
        # Generate audio
        audio = elevenlabs.generate(
            text=text,
            voice="XrExE9yKIg1WjnnlVkGX",
            model="eleven_monolingual_v1"
        )
        
        # Convert to BytesIO for streamlit audio playback
        audio_bytes = BytesIO()
        audio_bytes.write(audio)
        audio_bytes.seek(0)
        
        return audio_bytes
    except Exception as e:
        return None