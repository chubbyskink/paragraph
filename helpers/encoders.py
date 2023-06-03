import torch.nn as nn
from transformers import BertModel, BertTokenizer


class QuestionEncoder(nn.Module):
    def __init__(self):
        super(QuestionEncoder, self).__init__()

        # Load the pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def encode(self, question):
        # Tokenize the question
        tokens = self.tokenizer.encode_plus(
            question,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Forward pass through BERT model
        outputs = self.bert_model(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )

        # Extract the contextualized representations (outputs from BERT)
        contextualized_reps = outputs.last_hidden_state

        return contextualized_reps

# Used for both the paragraphs and answers
class ParagraphEncoder(nn.Module):
    def __init__(self):
        super(ParagraphEncoder, self).__init__()

        # Load the pre-trained BERT model
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, paragraph, answers):
        # Encode the paragraph using the BERT model
        paragraph_reps = self.bert_model(paragraph).last_hidden_state

        # Encode each answer and store the representations in a list
        answer_reps = []
        for answer in answers:
            answer_rep = self.bert_model(answer).last_hidden_state
            answer_reps.append(answer_rep)

        return paragraph_reps, answer_reps
