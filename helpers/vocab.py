from torchtext.vocab import build_vocab_from_iterator

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
def encode_text(text_data, vocab):
    encoded_data = [[vocab[token] for token in sentence.split()] for sentence in text_data]
    return encoded_data

def compute_vocab_size(questions, answers, paragraphs):
    all_text_data = questions + answers + paragraphs
    all_words = [word for sentence in all_text_data for word in sentence.split()]
    unique_words = set(all_words)
    vocab_size = len(unique_words)
    return vocab_size
