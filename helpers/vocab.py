import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

DATA_PATH = "/Users/gavinolsen/Desktop/code/data/"

# Step 1: Load and preprocess the data
def load_data(file_path):
    with open(DATA_PATH + file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    data = [line.strip() for line in data]
    return data

# Step 2: Build the vocabulary
def build_vocab(text_data):
    tokenized_data = [sentence.split() for sentence in text_data]
    vocab = build_vocab_from_iterator(tokenized_data, specials=['<unk>', '<pad>'], min_freq=1)
    return vocab

# Step 3: Encoding the data
def encode_text(text_data, vocab, max_length):
    encoded_data = [[vocab[token] for token in sentence.split()] for sentence in text_data]

    # Pad the sentences to a fixed length
    padded_data = pad_sequence([torch.tensor(sentence) for sentence in encoded_data], batch_first=True, padding_value=0)

    # Truncate or pad the sentences to the desired maximum length
    padded_data = padded_data[:, :max_length]

    return padded_data

def compute_vocab_size(questions, answers, paragraphs):
    all_text_data = questions + answers + paragraphs
    all_words = [word for sentence in all_text_data for word in sentence.split()]
    unique_words = set(all_words)
    vocab_size = len(unique_words)
    return vocab_size

def calculate_max_length(data):
    max_length = 0
    for sentence in data:
        length = len(sentence.split())
        if length > max_length:
            max_length = length
    return max_length