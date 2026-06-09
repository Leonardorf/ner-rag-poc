# src/llm_ollama.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"  # Opciones: llama3.2:3b | llama3.1:8b | phi3:mini | gemma2:2b


def ask_llama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)

    if response.status_code != 200:
        print("Error Ollama:", response.text)
        response.raise_for_status()

    return response.json()["response"]

