import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:0.6b"

def ask_ollama(prompt):
    """Send a prompt to Ollama and return the model's response."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["response"]

    except HTTPError as error:
        return f"Ollama API error: HTTP {error.code}"

    except URLError as error:
        return f"Could not connect to Ollama: {error.reason}"

    except (KeyError, json.JSONDecodeError):
        return "Error: Ollama returned an unexpected response."


if __name__ == "__main__": # Runs only when you directly execute:
    question = input("Enter your research question: ").strip()

    if not question:
        print("Error: Please enter a research question.")
        raise SystemExit(1)

    print("Sending question to Ollama...")
    answer = ask_ollama(question)

    print("\nAI Response:")
    print(answer)