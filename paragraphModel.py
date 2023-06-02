import torch
from transformers import BertForQuestionAnswering, BertTokenizer

path = "/Users/gavinolsen/Desktop/code/data/"

# Load the paragraph
f = open(path+"story.txt", "r")
paragraph = f.read()

# Load the questions
f = open(path+"questions.txt", "r")
questions = f.readlines()

# Load the answers
f = open(path+"answers.txt", "r")
answers = f.readlines()

# Create the question answering model
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

EPOCH_COUNT = 10

optimizer = torch.optim.AdamW(model.parameters())
for epoch in range(EPOCH_COUNT):
    total_loss = 0
    for question, answer in zip(questions, answers):
        # Tokenize the paragraph, question, and answer
        inputs = tokenizer.encode_plus(question, answer, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        # Predict the answer
        outputs = model(**inputs)
        loss = outputs.loss

   # Predict the answer
    outputs = model(**inputs)
    loss = outputs.loss

    # Update the model parameters
    optimizer.zero_grad()
    if loss is not None:
        loss.backward()
        total_loss += loss.item()
    
    optimizer.step()

    print(f"Epoch {epoch + 1} - Average Loss: {total_loss / len(questions)}")

# Save the model
model.save_pretrained("model")

