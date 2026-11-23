"""Stand-ins for the ONNX model and the Groq API, so tests are fast and offline."""


class DictTagger:
    """Tags a word B-<label> if it is in the dictionary. Multi-word entries are not needed in tests."""

    def __init__(self, vocab: dict[str, str]):
        self.vocab = {k.lower(): v for k, v in vocab.items()}
        self.calls: list[list[str]] = []

    def tag(self, words: list[str]) -> list[str]:
        self.calls.append(words)
        return [f"B-{self.vocab[w.lower()]}" if w.lower() in self.vocab else "O" for w in words]


class FakeLLM:
    """Returns a canned JSON reply and records every prompt it was sent."""

    def __init__(self, reply: dict | None = None):
        self.reply = reply or {
            "summary": "Strong Python background; no cloud experience.",
            "questions": [f"Question {i}?" for i in range(1, 6)],
        }
        self.calls: list[list[dict]] = []

    def chat_json(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        return self.reply
