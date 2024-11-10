import streamlit as st
from streamlit_mic_recorder import mic_recorder
import concurrent.futures
import base64
from PIL import Image
import os
from tempfile import NamedTemporaryFile
import google.generativeai as genai
from groq import Groq
from elevenlabs.client import ElevenLabs
from together import Together
from interpreter import interpreter
from duckduckgo_search import DDGS
# Initialize secrets


# Initialize clients
together_client = Together(api_key=TOGETHER_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Define reference models
reference_models = [
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    
]

# Helper functions
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

def image_to_data_url(image_path):
    """Convert image to base64 data URL."""
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_image}"

def extract_question_from_image(image_path):
    """Extract question from image using Gemini."""
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
    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [file]}]
    )
    response = chat_session.send_message(
        "extract the question and options as it is from the image. Note: Don't return any text (not even other single word)other than the Question and options. Make sure You extract the question as it is in the Image , Do not make Mistakes"
    )
    return response.text

def create_system_prompt(user_prompt):
    """Create system prompt for problem breakdown."""
    return f"""You are a mathematical genius and an expert teacher renowned for breaking down complex problems into clear, understandable steps. Your mission is to outline how one would approach dividing the following math problem into smaller sub-problems and describe the solving process.

**Instructions:**
- **Do not** provide the final answer, perform calculations, or actually solve the problem.
- **Focus** solely on how to divide the problem into sub-problems and describe the process one would follow to solve it.
- **Use** clear and precise language suitable for someone learning how to approach problem-solving.

**Math Problem:**
{user_prompt}

**Your breakdown should include:**

1. **Problem Division:**
   - Identify major components or sub-problems within the larger problem.

2. **Process Outline:**
   - Describe the general steps or the sequence of actions one would take for each sub-problem.

Format your response as:
[SUBDIVISION]
• Sub-problem 1: Description
• Sub-problem 2: Description
...

[PROCESS OUTLINE]
• For Sub-problem 1: Outline the process
• For Sub-problem 2: Outline the process
...
"""

def run_together_llm(model: str, prompt: str) -> str:
    """Runs the LLM model with the provided prompt."""
    response = together_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=3000,
    )
    return f"{model}: {response.choices[0].message.content}"



def solve_math_problem(user_prompt):
    """Main function to process and solve math problems."""
    breakdown_prompt = create_system_prompt(user_prompt)
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(reference_models)) as executor:
        future_to_model = {}
        for model in reference_models:
            if model.startswith("meta-llama"):
                future = executor.submit(run_together_llm, model, breakdown_prompt)
            else:
                raise ValueError(f"Unknown model type: {model}")
            future_to_model[future] = model

        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{model} generated an exception: {exc}')
    
    final_aggregator_prompt = f"""
    You have been provided with various strategies for dividing and outlining the solving process for a math problem from different models. Your task is to synthesize these strategies into a cohesive, logical sequence that outlines how one would approach dividing the problem into sub-problems and describing the solving process.

    Math Problem:
    {user_prompt}
    
    Strategies from models:
    {"\n\n".join(results)}

    Your goal is to combine these strategies into a single, well-structured explanation that focuses on how to break down the problem and the process one would follow to solve it, without actually solving or providing final answers.
    """
    
    # Replace DDGS.chat with Groq API call
    aggregated_strategy = groq_client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=[
            {"role": "user", "content": final_aggregator_prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    ).choices[0].message.content
    
    interpreter.system_message = """
    You have been provided with strategies for the solving process for a math problem. Your task is to solve the problem a single step at a time and if that particular step needs to do some computations write code and execute it and make sure you do one step at a time and if need to compute something write correct code for that current step and execute it and then continue to next step and make sure you do not skip any step and do not provide the final answer until you reach the final step.
    """
    interpreter.auto_run = True
    interpreter.llm.model = "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    interpreter.llm.api_key = TOGETHER_API_KEY
    solution = interpreter.chat(user_prompt + "\n" + aggregated_strategy)
    return solution

def transcribe_audio(audio_bytes, filename):
    """Transcribe audio using Groq."""
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
        return transcription.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def stream_audio_response(text):
    """Generate audio response using ElevenLabs API."""
    try:
        if not text:
            return None
            
        audio_stream = elevenlabs_client.generate(
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

def process_image(uploaded_file):
    """Process the uploaded image and extract text."""
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
    """Process the math problem and generate solution."""
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
    """Display the conversation history with audio players."""
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
    """Process voice input and generate response."""
    try:
        transcribed_query = transcribe_audio(audio['bytes'], "audio.wav")
        
        st.session_state.conversation.append(
            {"role": "user", "content": transcribed_query}
        )

        api_messages = [
            {"role": "system", "content": f"Question: {st.session_state.question}\nSolution: {st.session_state.step_by_step_solution}"}
        ]
        
        for msg in st.session_state.conversation:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        completion = groq_client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=api_messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        assistant_response = completion.choices[0].message.content
        audio_base64 = stream_audio_response(assistant_response)
        
        st.session_state.conversation.append({
            "role": "assistant", 
            "content": assistant_response,
            "audio_base64": audio_base64
        })
        
        st.rerun()

    except Exception as e:
        st.error(f"Error processing voice input: {str(e)}")

def main():
    # Page configuration
    st.set_page_config(
        page_title="Math Problem Solver with Voice Interaction",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main { padding: 2rem; color: #1a1a1a; }
        .question-box { background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #d1d1d1; color: #1a1a1a; }
        .solution-box { background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #b8daff; color: #1a1a1a; }
        .chat-message { margin: 0.5rem 0; padding: 1rem; border-radius: 0.5rem; border: 1px solid #d1d1d1; color: #1a1a1a; }
        .user-message { background-color: #ffffff; margin-right: 2rem; }
        .assistant-message { background-color: #ffffff; margin-left: 2rem; }
        .conversation-area { padding: 1rem; border-radius: 0.5rem; border: 1px solid #d1d1d1; margin-bottom: 1rem; max-height: 500px; overflow-y: auto; background-color: #ffffff; }
        #the-title { text-align: center; font-family: 'Arial', sans-serif; font-size: 36px; font-weight: bold; }
        #the-tagline { text-align: center; font-family: 'Arial', sans-serif; font-size: 24px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

    # Title and tagline
    st.markdown('<h1 id="the-title">🧮 EduLlama</h1>', unsafe_allow_html=True)
    st.markdown('<h2 id="the-tagline">Learn Math Smarter, Not Harder!</h2>', unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Create main columns
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Problem Input")
        tabs = st.tabs(["📸 Image Input", "✍️ Text Input"])
        
        # Image Input Tab
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

        # Text Input Tab
        with tabs[1]:
            user_text = st.text_area("Type your math problem here", height=150)
            if st.button("🧮 Solve Problem", key="solve_text") and user_text:
                process_problem(user_text)

        # Display solution if available
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

    # Voice Assistant Column
    with col2:
        st.header("🎙️ Voice Assistant")
        
        # Conversation area
        with st.container():
            st.markdown('<div class="conversation-area">', unsafe_allow_html=True)
            display_conversation()
            st.markdown('</div>', unsafe_allow_html=True)

        # Voice controls
        with st.container():
            vcol1, vcol2 = st.columns([3, 1])
            with vcol1:
                audio = mic_recorder(
                    start_prompt="Start Speaking",
                    stop_prompt="Stop Recording",
                    just_once=True,
                    key=f"voice_recorder_{st.session_state.audio_key}"
                )
            with vcol2:
                if st.button("Clear Chat"):
                    st.session_state.conversation = []
                    st.session_state.audio_key = 0
                    st.rerun()
        
        # Process voice input if available
        if audio:
            process_voice_input(audio)
            st.session_state.audio_key += 1

if __name__ == "__main__":
    main()
