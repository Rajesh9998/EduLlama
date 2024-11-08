import concurrent.futures
from together import Together
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from interpreter import interpreter


load_dotenv()
# Initialize Together client
client = Together(api_key=os.environ["TOGETHER_API_KEY"])


# Define models used for reasoning
reference_models = [
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    #"meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "gpt-4o-mini",
    "claude-3-haiku"
]

# User's math problem
user_prompt = """

The value of √(7 + √(7 - √(7 + √7 - ... ∞ is
(a) 5
(b) 4
(c) 3
(d) 2


"""

# Define the system prompt to instruct models to reason through the problem step by step
aggregator_system_prompt = f"""You are a mathematical genius and an expert teacher renowned for breaking down complex problems into clear, understandable steps. Your mission is to outline how one would approach dividing the following math problem into smaller sub-problems and describe the solving process.

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
    print(f"{model}: {response.choices[0].message.content}")
    return f"{model}: {response.choices[0].message.content}"

def run_ddg_chat(model: str, prompt: str) -> str:
    """Runs the prompt through DuckDuckGo chat."""
    ddgs = DDGS()
    response = ddgs.chat(prompt, model=model)  # Use specified model
    print(f"DuckDuckGo {model}: {response}")
    return f"DuckDuckGo {model}: {response}"


def main():
    # Generate the detailed system prompt for breaking down the math problem
    breakdown_prompt = aggregator_system_prompt.format(user_prompt=user_prompt)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(reference_models)) as executor:
        future_to_model = {}  # Use a dictionary to map futures to models
        for model in reference_models:
            if model.startswith("meta-llama"):
                future = executor.submit(run_together_llm, model, breakdown_prompt)
            elif model in ["gpt-4o-mini", "claude-3-haiku"]: # Check for DDG models
                future = executor.submit(run_ddg_chat, model, breakdown_prompt)
            else:
                 raise ValueError(f"Unknown model type: {model}") # Handle other cases

            future_to_model[future] = model # Store future and model

        results = []
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]  # Get the model from the dictionary
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{model} generated an exception: {exc}')
    
    # Combine the responses from the models into a final aggregated explanation
    final_aggregator_prompt = """
    You have been provided with various strategies for dividing and outlining the solving process for a math problem from different models. Your task is to synthesize these strategies into a cohesive, logical sequence that outlines how one would approach dividing the problem into sub-problems and describing the solving process.

    Math Problem:
    {user_prompt}
    
    Strategies from models:
    {responses}

    Your goal is to combine these strategies into a single, well-structured explanation that focuses on how to break down the problem and the process one would follow to solve it, without actually solving or providing final answers.
    """
    
    # Aggregate the model responses into a single explanation
    aggregator_prompt = final_aggregator_prompt.format(user_prompt=user_prompt, responses="\n\n".join(results))
    """
    # Get the final aggregated explanation
    final_response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",  # Can use a larger model here for final aggregation
        messages=[{"role": "system", "content": aggregator_prompt}],
        temperature=0.4,
        max_tokens=4500,
    )
    print("\n\nFinal Aggregated Explanation:")
    print(final_response.choices[0].message.content)

    """
    final_response= DDGS().chat(user_prompt+"\n"+aggregator_prompt, model="gpt-4o-mini")
    print("Final Response by DuckDuckGo GPT-4o-Mini: ", final_response)
    interpreter.system_message = """
       
    You have been provided with strategiey for the solving process for a math problem. Your task is to solve the problem a single step at a time and if that particular step needs to do some computations write code and exicute it and make sure you do one step at a time and if need to compute something write correct code for that current step and exicute it and then continue to next step and make sure you do not skip any step and do not provide the final answer until you reach the final step.
    """
    interpreter.auto_run = True
    interpreter.llm.model = "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    interpreter.chat(user_prompt+"\n"+final_response)


if __name__ == "__main__":
    main()
