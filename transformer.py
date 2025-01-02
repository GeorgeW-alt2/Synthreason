import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os
import json

# Constants
KB_MEMORY_UNCOMPRESSED = 10000
SEQUENCE_LENGTH = 3
NUM_EPOCHS = 10
GENERATE_LENGTH = 1000
TEMPERATURE = 0.7
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_DIRECTORY = "saved_models"

class TextGenerator(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TextGenerator, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def save_model(model, word_to_index, index_to_word, model_name, input_size, directory=MODEL_DIRECTORY):
    """Save the model, vocabulary mappings, and model configuration."""
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(directory, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save vocabulary mappings and model configuration
    vocab_path = os.path.join(directory, f"{model_name}_config.json")
    config_data = {
        'word_to_index': word_to_index,
        'index_to_word': {str(k): v for k, v in index_to_word.items()},  # Convert int keys to strings for JSON
        'input_size': input_size
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    print(f"Model and configuration saved to {directory}")

def load_model(model_name, hidden_sizes, directory=MODEL_DIRECTORY):
    """Load the model, vocabulary mappings, and model configuration."""
    model_path = os.path.join(directory, f"{model_name}_model.pth")
    config_path = os.path.join(directory, f"{model_name}_config.json")
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Model or configuration files not found in {directory}")
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    word_to_index = config_data['word_to_index']
    index_to_word = {int(k): v for k, v in config_data['index_to_word'].items()}  # Convert string keys back to int
    input_size = config_data['input_size']
    vocab_size = len(word_to_index)
    
    # Create and load model
    model = TextGenerator(input_size, hidden_sizes, vocab_size)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    
    return model, word_to_index, index_to_word, vocab_size

def create_dataset_from_text(text, sequence_length):
    # Tokenize and create vocabulary
    words = text.lower().split()
    vocab = sorted(set(words))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    vocab_size = len(vocab)

    # Create sequences
    sequences = []
    next_words = []
    
    for i in range(len(words) - sequence_length):
        sequences.append([word_to_index[words[i + j]] for j in range(sequence_length)])
        next_words.append(word_to_index[words[i + sequence_length]])

    # Convert to PyTorch tensors
    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(next_words, dtype=torch.long)
    
    # Create one-hot encoded input
    X_one_hot = F.one_hot(X, num_classes=vocab_size).float()
    X_one_hot = X_one_hot.reshape(X_one_hot.shape[0], -1)  # Flatten the sequence dimension

    return X_one_hot, y, word_to_index, index_to_word, vocab_size

def train_model(model, data_loader, num_epochs, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Training on {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def generate_text(model, seed_text, word_to_index, index_to_word, vocab_size, sequence_length, 
                 num_words, temperature, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model = model.to(device)
    
    # Process seed text
    words = seed_text.lower().split()
    if len(words) < sequence_length:
        raise ValueError(f"Seed text must contain at least {sequence_length} words")
    
    current_sequence = words[-sequence_length:]
    generated_words = []
    
    with torch.no_grad():
        for _ in range(num_words):
            # Convert current sequence to tensor
            try:
                sequence_indices = [word_to_index[word] for word in current_sequence]
            except KeyError:
                print("Warning: Unknown word in seed text. Using random word from vocabulary.")
                sequence_indices = np.random.choice(list(index_to_word.keys()), sequence_length).tolist()
            
            sequence_tensor = torch.tensor([sequence_indices], dtype=torch.long)
            x = F.one_hot(sequence_tensor, num_classes=vocab_size).float()
            x = x.reshape(1, -1).to(device)
            
            # Get predictions
            logits = model(x)
            
            # Apply temperature
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Sample from the distribution
            next_word_idx = torch.multinomial(probs, 1).item()
            next_word = index_to_word[next_word_idx]
            
            generated_words.append(next_word)
            current_sequence = current_sequence[1:] + [next_word]
    
    return ' '.join(generated_words)

def main():
    model_name = "text_generator_v1"
    hidden_sizes = [512, 256]
    
    filename = 'test.txt'
    # First, try to load the training data to get vocabulary size
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = ' '.join(file.read().split()[:KB_MEMORY_UNCOMPRESSED])
    except FileNotFoundError:
        print("Error: {filename} not found")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    
    
    # Try to load saved model
    try:
        print("Attempting to load saved model...")
        model, word_to_index, index_to_word, vocab_size = load_model(
            model_name,
            hidden_sizes=hidden_sizes
        )
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("No saved model found. Training new model...")
        # Create dataset
        print("Creating dataset...")
        X, y, word_to_index, index_to_word, vocab_size = create_dataset_from_text(text, SEQUENCE_LENGTH)
        input_size = X.shape[1]
        # Create model
        model = TextGenerator(input_size, hidden_sizes, vocab_size)
        
        # Create data loader
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train model
        print("Training model...")
        train_model(model, data_loader, NUM_EPOCHS, LEARNING_RATE)
        
        # Save the trained model
        save_model(model, word_to_index, index_to_word, model_name, input_size)
    
    while True:
        seed_text = input("User:")
        try:
            generated_text = generate_text(
                model, 
                seed_text, 
                word_to_index, 
                index_to_word, 
                vocab_size, 
                SEQUENCE_LENGTH,
                num_words=250,
                temperature=TEMPERATURE
            )
            print(f"\nSeed text: {seed_text}")
            print(f"Generated text: {generated_text}")
        except Exception as e:
            print(f"Error generating text: {e}")

if __name__ == "__main__":
    main()