import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import ollama

load_dotenv()  # reads GEMINI_API_KEY from .env

TEMPERATURE = 0.7  # same setting as Irugalbandara et al. (2024)
GEMINI_MODEL = "gemini-3.5-flash-lite"
LOCAL_MODEL = "mistral"

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_public(prompt, max_retries=5):
    """Public API model (Gemini), retrying if Google's servers are busy."""
    for attempt in range(1, max_retries + 1):
        start = time.perf_counter()
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except Exception as error:
            print(f"Attempt {attempt} failed: {error}")
            if attempt == max_retries:
                raise
            time.sleep(10 * attempt)  # wait 10s, 20s, 30s... before retrying
            continue
        latency = time.perf_counter() - start
        usage = response.usage_metadata
        output_tokens = usage.candidates_token_count or 0
        return {
            "deployment": "public",
            "model": GEMINI_MODEL,
            "response": response.text,
            "input_tokens": usage.prompt_token_count,
            "output_tokens": output_tokens,
            "thinking_tokens": usage.thoughts_token_count or 0,
            "latency_s": latency,
            "latency_per_token_s": latency / output_tokens if output_tokens else None,
            "attempts": attempt,
        }


def generate_private(prompt):
    """Private self-hosted model (Mistral 7B through Ollama)."""
    start = time.perf_counter()
    response = ollama.generate(
        model=LOCAL_MODEL,
        prompt=prompt,
        options={"temperature": TEMPERATURE},
    )
    latency = time.perf_counter() - start
    output_tokens = response["eval_count"]
    # Same keys in the same order as generate_public, so CSV columns line up
    return {
        "deployment": "private",
        "model": LOCAL_MODEL,
        "response": response["response"],
        "input_tokens": response["prompt_eval_count"],
        "output_tokens": output_tokens,
        "thinking_tokens": 0,
        "latency_s": latency,
        "latency_per_token_s": latency / output_tokens if output_tokens else None,
        "attempts": 1,
    }


if __name__ == "__main__":
    test = "In two sentences, what is a customer support ticket?"
    results = (generate_public(test), generate_private(test))
    for result in results:
        print(result["deployment"], "|", round(result["latency_s"], 2), "s |",
              result["output_tokens"], "tokens")
        print(result["response"], "\n")
    # Check that both models return the same columns in the same order
    print("Same columns:", list(results[0].keys()) == list(results[1].keys()))