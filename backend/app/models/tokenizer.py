import json


class BPETokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)

    def encode(self, text):
        return [self.vocab.get(word, 0) for word in text.lower().split()]

    def decode(self, tokens):
        return " ".join([self.inv_vocab.get(token, "<unk>") for token in tokens])