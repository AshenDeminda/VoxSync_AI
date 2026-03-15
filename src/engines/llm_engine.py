import ollama
import src.config as config

class LLMEngine:
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.history = [
            {"role": "system", "content": "You are a helpful voice assistant. Keep answers brief and conversational."}
        ]

    def chat(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        response = ollama.chat(model=self.model, messages=self.history)
        reply = response['message']['content']
        self.history.append({"role": "assistant", "content": reply})
        return reply