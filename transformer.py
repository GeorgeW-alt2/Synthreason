#Spiking neural network (SNN) 7.31 - George W - 24,12,2024
import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Constants
KB_MEMORY_UNCOMPRESSED = 50000
n = 4  # Use quadgrams for training
num_epochs = 10
generate_length = 1000
temperature = 0.3
feedforward_enhancer = KB_MEMORY_UNCOMPRESSED
# Preprocessing and Vocabulary
def preprocess_text(text):
    """Clean and tokenize text."""
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()[:KB_MEMORY_UNCOMPRESSED]
    return [word for word in tokens if len(word) > 1 or word in {"i", "a"}]

def build_vocabulary(text_data):
    """Build vocabulary with word frequencies."""
    tokens = preprocess_text(text_data)
    word_counts = {word: tokens.count(word) for word in set(tokens)}
    if tokens:  # Ensure the tokens list is not empty
        last_word = tokens[-1]
        word_counts[last_word] += feedforward_enhancer
       


    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    return word_to_index, len(vocab)

def create_sequences(word_to_index, text, sequence_length):
    """Convert text into sequences."""
    # Encode the text using the word-to-index mapping
    encoded = [word_to_index[word] for word in text if word in word_to_index]
    
    # Create sequences of the specified length
    return [(encoded[i-sequence_length:i], encoded[i]) for i in range(sequence_length, len(encoded))]


# Dataset Class
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Knowledge-Augmented LSTM Model
class KANEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, knowledge_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.knowledge_embedding = nn.Embedding(vocab_size, knowledge_dim)

    def forward(self, x):
        return torch.cat((self.word_embedding(x), self.knowledge_embedding(x)), dim=-1)
import torch
from torch.autograd import Function

class CustomLossFunction(Function):
    @staticmethod
    def forward(ctx, outputs, targets):
        """
        Forward pass for custom loss.
        Save variables for backward computation using `ctx.save_for_backward`.
        """
        ctx.save_for_backward(outputs, targets)
        # Example: Custom loss calculation (e.g., mean squared error)
        loss = (outputs.min() * targets.min() / -grad_output) / (outputs.min() * targets.max()) + (outputs.min() * targets)*(-outputs / outputs.min() * targets.max()) / (outputs.min() * targets)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for custom gradients.
        Use saved variables to compute the gradient of the loss with respect to inputs.
        """
        outputs, targets = ctx.saved_tensors
        grad_outputs = (outputs.min() * targets.min() / -grad_output) / (outputs.min() * targets.max()) + (outputs.min() * targets)*(-outputs / outputs.min() * targets.max()) / (outputs.min() * targets)
        # Multiply by the incoming gradient (chain rule)
        grad_outputs *= grad_output
        return grad_outputs, None  # Return gradients for all inputs to forward
class KnowledgeAugmentedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=150, knowledge_dim=100, rnn_units=386, dropout_rate=0.4):
        super().__init__()
        self.embedding = KANEmbedding(vocab_size, embedding_dim, knowledge_dim)
        self.lstm = nn.LSTM(embedding_dim + knowledge_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))
        
def train_model(model, data_loader, num_epochs, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Debug shapes

            optimizer.zero_grad()
            outputs = model(inputs)

            # Ensure output shape matches target shape
            assert outputs.shape[0] == targets.shape[0], "Batch sizes do not match."
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Save and Load Functions
def save_model_and_vocab(model, word_to_index):
    torch.save(model.state_dict(), 'knowledge_augmented_lstm.mdl')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    print("Model and vocabulary saved.")

def load_model_and_vocab(vocab_path='vocab.pkl', model_path='knowledge_augmented_lstm.mdl'):
    with open(vocab_path, 'rb') as f:
        word_to_index = pickle.load(f)
    vocab_size = len(word_to_index)
    model = KnowledgeAugmentedLSTM(vocab_size)
    model.load_state_dict(torch.load(model_path, weights_only= True))
    model.eval()
    print("Model and vocabulary loaded.")
    return model, word_to_index

# Text Generation
def generate_text(model, word_to_index, input_text, sequence_length, generate_length, temperature):
    input_sequence = preprocess_text(input_text)
    indices = [word_to_index.get(word, -1) for word in input_sequence if word in word_to_index]

    if not indices:
        return "Input text contains no recognizable words."

    generated_text = []
    input_tensor = torch.tensor(indices[-sequence_length:], dtype=torch.long).unsqueeze(0)

    for _ in range(generate_length):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output / temperature, dim=1).squeeze()
            next_word_idx = torch.multinomial(probabilities, 1).item()
            generated_text.append(next_word_idx)

            input_tensor = torch.cat((input_tensor[:, 1:next_word_idx], torch.tensor([[next_word_idx]])), dim=1)

    reverse_vocab = {i: word for word, i in word_to_index.items()}
    return ' '.join([reverse_vocab.get(idx, "<UNK>") for idx in generated_text])

# Main Function
def main():
    choice = input("Do you want to (1) train or (2) load a model: ")

    if choice == '1':
        with open("test.txt", encoding="utf-8") as f:
            text = f.read().lower()

        word_to_index, vocab_size = build_vocabulary(text)
        sequences = create_sequences(word_to_index, preprocess_text(text), sequence_length=4)
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        model = KnowledgeAugmentedLSTM(vocab_size)
        train_model(model, data_loader, num_epochs=num_epochs)
        save_model_and_vocab(model, word_to_index)
    elif choice == '2':
        model, word_to_index = load_model_and_vocab()
    else:
        print("Invalid option.")
        return

    while True:
        user_input = input("User: ")
        print("AI:", generate_text(model, word_to_index, generate_text(model, word_to_index, user_input, sequence_length=4, generate_length=generate_length, temperature=temperature), sequence_length=4, generate_length=generate_length, temperature=temperature))

if __name__ == "__main__":
    main()