import torch
from torch import nn

# Define the decoder module
class AnswerDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(AnswerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoded_paragraph, encoded_question):
        # Combine the encoded representations of the paragraph and question
        combined = torch.cat((encoded_paragraph, encoded_question), dim=1)

        # Pass the combined representation through the decoder linear layer
        output = self.fc(combined)
        return output