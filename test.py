import concurrent.futures
from together import Together
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from interpreter import interpreter
import google.generativeai as genai

load_dotenv()
# Initialize Together and Gemini clients
client = Together(api_key=os.environ["TOGETHER_API_KEY"])
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Define models used for reasoning
reference_models = [
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "gpt-4o-mini",
    "claude-3-haiku"
    
]

def extract_question_from_image(image_path):
    """Extract question and options from image using Gemini."""
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
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=3000,

    )
    return f"{model}: {response.choices[0].message.content}"

def run_ddg_chat(model: str, prompt: str) -> str:
    """Runs the prompt through DuckDuckGo chat."""
    ddgs = DDGS()
    response = ddgs.chat(prompt, model=model)
    return f"DuckDuckGo {model}: {response}"

def solve_math_problem(user_prompt):
    """Main function to process and solve math problems."""
    # Generate the detailed system prompt for breaking down the math problem
    breakdown_prompt = create_system_prompt(user_prompt)
    
    # Collect model responses
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(reference_models)) as executor:
        future_to_model = {}
        for model in reference_models:
            if model.startswith("meta-llama"):
                future = executor.submit(run_together_llm, model, breakdown_prompt)
            elif model in ["gpt-4o-mini", "claude-3-haiku"]:
                future = executor.submit(run_ddg_chat, model, breakdown_prompt)
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
    
    # Create final aggregator prompt
    final_aggregator_prompt = f"""
    You have been provided with various strategies for dividing and outlining the solving process for a math problem from different models. Your task is to synthesize these strategies into a cohesive, logical sequence that outlines how one would approach dividing the problem into sub-problems and describing the solving process.

    Math Problem:
    {user_prompt}
    
    Strategies from models:
    {"\n\n".join(results)}

    Your goal is to combine these strategies into a single, well-structured explanation that focuses on how to break down the problem and the process one would follow to solve it, without actually solving or providing final answers.
    """
    
    # Get final solution using DuckDuckGo GPT-4
    ddgs = DDGS()
    aggregated_strategy = ddgs.chat(user_prompt + "\n" + final_aggregator_prompt, model="gpt-4o-mini")
    
    # Set up interpreter for step-by-step solution
    interpreter.system_message = """
    You have been provided with strategies for the solving process for a math problem. Your task is to solve the problem a single step at a time and if that particular step needs to do some computations write code and execute it and make sure you do one step at a time and if need to compute something write correct code for that current step and execute it and then continue to next step and make sure you do not skip any step and do not provide the final answer until you reach the final step.
    """
    interpreter.auto_run = True
    interpreter.llm.model = "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    
    # Get final solution
    solution = interpreter.chat(user_prompt + "\n" + aggregated_strategy)
    return solution

