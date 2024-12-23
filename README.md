# EduLlama 🦙 
>Your AI-Powered Math Companion that can help you with JEE Math Problems





## Overview

EduLlama is an interactive, AI-powered math tutor designed to help students master complex mathematical problems, particularly at the JEE level. By combining multiple Large Language Models and advanced AI technologies, EduLlama provides step-by-step solutions and interactive voice assistance for a personalized learning experience.

![](Examples/Screenshot%202024-11-11%20024742.png)

## Features

- 🤖 **Multiple LLM Integration**: Utilizes Meta's Llama 3.1 and 3.2 models through Together AI's Mixture of Agents (MoA) architecture
- 📷 **Image Problem Input**: Upload images of math problems for automatic text extraction using Google Gemini
- 🎯 **Precise Calculations**: Integration with Open Interpreter for accurate mathematical computations
- 🗣️ **Interactive Voice Assistant**: Real-time voice interactions powered by Groq and ElevenLabs
- 📝 **Step-by-Step Solutions**: Detailed breakdown of complex problems into manageable steps
- 🔄 **Real-time Processing**: Fast inference using Together AI's high-speed APIs

## Technology Stack

- **LLM Models**: Meta's Llama 3.1 & 3.2 (via Together AI)
- **Speech-to-Text**: Whisper (via Groq)
- **Text-to-Speech**: ElevenLabs
- **Image Processing**: Google Gemini
- **Framework**: Streamlit
- **Additional Tools**: Open Interpreter

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/edullama.git
cd edullama
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.streamlit/secrets.toml`:
```toml
TOGETHER_API_KEY = "your_together_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
GROQ_API_KEY = "your_groq_api_key"
ELEVENLABS_API_KEY = "your_elevenlabs_api_key"
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Upload Problem**:
   - Use the image upload feature to submit a picture of your math problem, or
   - Type your problem directly into the text input area

2. **Get Solution**:
   - Click "Solve Problem" to receive a detailed step-by-step solution
   - Review the breakdown and explanation of each step

3. **Voice Interaction**:
   - Use the voice assistant to ask questions about the solution
   - Receive verbal explanations and clarifications
   - Clear chat history as needed

## API Requirements

To run EduLlama, you'll need API keys from:
- [Together AI](https://together.ai)
- [Google Cloud (Gemini)](https://cloud.google.com)
- [Groq](https://groq.com)
- [ElevenLabs](https://elevenlabs.io)

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Team T-REX for the development and implementation
- Meta for the Llama models
- Together AI for their MoA architecture
- All other technology partners and contributors

## Contact

For questions and feedback, please open an issue in the GitHub repository or contact the maintainers directly.

---
Made with ❤️ by Rajesh
