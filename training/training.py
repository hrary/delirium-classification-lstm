from models.classificationAlgorithm import ClassificationAlgorithm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.datasets.VitalsDataset import VitalsDataset

input_dim = 12  # Number of features in the input data
hidden_dim = 32  # Number of features in the hidden state of the LSTM
num_layers = 2  # Number of stacked LSTM layers
output_dim = 1  # Binary classification

sequence_length = 64  # Length of each input sequence
batch_size = 32 # Number of sequences in each batch
learning_rate = 1e-3 # Learning rate for the optimizer
epochs = 5 # Number of training epochs

train_dataset = VitalsDataset("data/somedatahere")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassificationAlgorithm(input_dim, hidden_dim, num_layers, output_dim).to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(sequences).squeeze(-1)
        loss = loss_function(outputs, labels)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


torch.save(model.state_dict(), "model/delirium_lstm.pth")