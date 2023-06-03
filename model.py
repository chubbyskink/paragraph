import torch
from torch import optim, nn
from helpers.vocab import load_data, build_vocab, encode_text, compute_vocab_size
from helpers.loader import create_dataloader
from helpers.qamodel import QuestionAnsweringModel

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

paragraphs = load_data('paragraphs.txt')
questions = load_data('questions.txt')
answers = load_data('answers.txt')

paragraph_vocab = build_vocab(paragraphs)
question_vocab = build_vocab(questions)
answer_vocab = build_vocab(answers)

encoded_paragraphs = encode_text(paragraphs, paragraph_vocab)
encoded_questions = encode_text(questions, question_vocab)
encoded_answers = encode_text(answers, answer_vocab)

vocab_size = compute_vocab_size(questions, answers, paragraphs)

print("Total vocabulary: ", vocab_size)

# Instantiate the model
embedding_dim = 100  # Replace with the desired embedding dimension
hidden_dim = 256  # Replace with the desired hidden dimension
output_dim = 2  # Replace with the desired output dimension (e.g., start and end positions)
model = QuestionAnsweringModel(hidden_dim, output_dim)

# Define your loss function
loss_function = nn.CrossEntropyLoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define data loader
batch_size = 32
data_loader = create_dataloader(encoded_paragraphs, encoded_questions, encoded_answers, batch_size)

num_epochs = 2

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        # Clear gradients
        optimizer.zero_grad()

        # Get batch inputs (paragraphs, questions, answers)
        paragraphs, questions, answers = batch

        # Forward pass
        answer_logits = model(paragraphs, questions)

        # Compute loss
        loss = loss_function(answer_logits, answers)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Print loss or other metrics if desired
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    
    # Save the model every 100 epochs
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
