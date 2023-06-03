import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class QaData(Dataset):
    def __init__(self, encoded_paragraphs, encoded_questions, encoded_answers):
        self.encoded_paragraphs = encoded_paragraphs
        self.encoded_questions = encoded_questions
        self.encoded_answers = encoded_answers
    
    def __len__(self):
        return len(self.encoded_paragraphs)
    
    def __getitem__(self, idx):
        encoded_paragraph = self.encoded_paragraphs[idx]
        encoded_question = self.encoded_questions[idx]
        encoded_answer = self.encoded_answers[idx]
        
        return encoded_paragraph, encoded_question, encoded_answer


def create_dataloader(encoded_paragraphs, encoded_questions, encoded_answers, batch_size):
    # Convert lists to tensors
    tensor_paragraphs = torch.tensor(encoded_paragraphs)
    tensor_questions = torch.tensor(encoded_questions)
    tensor_answers = torch.tensor(encoded_answers)

    # Pad the sequences to make them of equal length within each batch
    padded_paragraphs = pad_sequence(tensor_paragraphs, batch_first=True)
    padded_questions = pad_sequence(tensor_questions, batch_first=True)
    padded_answers = pad_sequence(tensor_answers, batch_first=True)

    # Create the dataset
    dataset = QaData(padded_paragraphs, padded_questions, padded_answers)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
