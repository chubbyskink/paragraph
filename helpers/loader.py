from torch.utils.data import Dataset, DataLoader

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
    # Create the dataset
    dataset = QaData(encoded_paragraphs, encoded_questions, encoded_answers)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
