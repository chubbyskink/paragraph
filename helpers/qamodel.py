from torch import nn
# from encoders import ParagraphEncoder, QuestionEncoder
from helpers.encoders import ParagraphEncoder, QuestionEncoder
from helpers.decoders import AnswerDecoder

# Define your model
class QuestionAnsweringModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(QuestionAnsweringModel, self).__init__()
        self.paragraph_encoder = ParagraphEncoder()
        self.question_encoder = QuestionEncoder()
        self.decoder = AnswerDecoder(hidden_dim * 2, output_dim)  # Multiply by 2 because we concatenate paragraph and question

    # Allow answers to be given here
    def forward(self, paragraph, question, answers=None):
        # Encode the paragraph and question
        encoded_paragraph = self.paragraph_encoder(paragraph)
        encoded_question = self.question_encoder(question)

        # Decode to get the answer
        answer_logits = self.decoder(encoded_paragraph, encoded_question)

        return answer_logits