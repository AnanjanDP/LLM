class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(self.tokenize(text))
        tokens = sorted(list(tokens))
        self.token_to_id = {tok: i for i, tok in enumerate(tokens, start=1)} 
        self.token_to_id['<PAD>'] = 0  
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def tokenize(self, text):
        return text.lower().split()

    def encode(self, text):
        return [self.token_to_id.get(tok, 0) for tok in self.tokenize(text)]  

    def decode(self, ids):
        return " ".join([self.id_to_token.get(i, "<UNK>") for i in ids])

if __name__ == "__main__":
    texts = [
        "Hello how are you",
        "I am gay",
        "How about a game of chess"
    ]
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts)
    sample = "I am gay"
    encoded = tokenizer.encode(sample)
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
