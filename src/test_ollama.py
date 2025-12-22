import requests

payload = {
    "model": "llama3.1:8b",
    "prompt": "Decí hola en español",
    "stream": False
}

r = requests.post("http://localhost:11434/api/generate", json=payload)
print(r.status_code)
print(r.text)
