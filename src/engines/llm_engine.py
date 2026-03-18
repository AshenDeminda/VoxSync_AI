import ollama
import src.config as config


class OllamaUnavailableError(RuntimeError):
    pass

class LLMEngine:
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.host = getattr(config, "OLLAMA_HOST", None)
        self.client = ollama.Client(host=self.host) if self.host else ollama.Client()
        self.history = [
            {"role": "system", "content": "You are a helpful voice assistant. Keep answers brief and conversational."}
        ]

    def chat(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        try:
            response = self.client.chat(model=self.model, messages=self.history)
        except ConnectionError as e:
            # Roll back the last user message so history stays consistent.
            self.history.pop()
            host_hint = self.host or "http://127.0.0.1:11434"
            raise OllamaUnavailableError(
                f"Failed to connect to Ollama at {host_hint}. Start Ollama (or run `ollama serve`) and try again."
            ) from e
        except ollama.ResponseError as e:
            self.history.pop()
            raise RuntimeError(f"Ollama error: {e}") from e

        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply