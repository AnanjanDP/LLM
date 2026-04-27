import torch
from app.models.model import TransformerModel
from app.models.tokenizer import BPETokenizer

device = torch.device("cpu")

# ✅ Load tokenizer
tokenizer = BPETokenizer.load("app/models/tokenizer.json")

vocab_size = len(tokenizer.vocab)

# ✅ Initialize model
model = TransformerModel(vocab_size).to(device)

# ✅ Load weights if available
try:
    model.load_state_dict(torch.load("app/models/model.pth", map_location=device))
    print("✅ Model loaded")
except:
    print("⚠️ model.pth not found, using untrained model")

model.eval()


def generate_stream(query, max_len=20):
    tokens = tokenizer.encode(query)   # list[int]

    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    # shape → [1, seq]

    generated = tokens.copy()

    for _ in range(max_len):
        outputs = model(input_ids)  # [1, seq, vocab]

        next_token_logits = outputs[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()

        generated.append(next_token)

        input_ids = torch.tensor([generated], dtype=torch.long).to(device)

    return tokenizer.decode(generated)