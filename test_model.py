import torch
from tokenizer import Tokenizer
from model import TransformerBlock

def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def pad_sequences(sequences, pad_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded)

if __name__ == "__main__":
    path = "wizard_of_oz.txt"
    dataset = load_text_file(path)
    print(f"Loaded {len(dataset)} lines from {path}")
    print(f"First line: {dataset[0]}")

    tokenizer = Tokenizer()
    tokenizer.build_vocab(dataset)
    print(f"Vocab size: {len(tokenizer.token_to_id)}")

    encoded_data = [tokenizer.encode(sentence) for sentence in dataset]
    print(f"Encoded first sentence: {encoded_data[0]}")

    input_tensor = pad_sequences(encoded_data)
    print(f"Padded input tensor shape: {input_tensor.shape}")

    embed_size = 512
    vocab_size = len(tokenizer.token_to_id)
    embedding = torch.nn.Embedding(vocab_size, embed_size)

    x = embedding(input_tensor)
    print(f"Embedding output shape: {x.shape}")

    heads = 8
    dropout = 0.1
    forward_expansion = 4

    block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
    out = block(x, x, x, mask=None)
    print(f"Transformer block output shape: {out.shape}")
