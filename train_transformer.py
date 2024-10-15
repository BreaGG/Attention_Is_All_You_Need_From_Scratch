from torch.utils.data import DataLoader
from data_preparation import prepare_data
from transformer_model import Transformer
import torch
import torch.nn as nn
import torch.optim as optim

# Training loop
def train(model, data_loader, epochs, criterion, optimizer):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for src_batch, tgt_batch in data_loader:
            src_input = src_batch
            tgt_input = tgt_batch[:, :-1]  # Remove the last token for input
            tgt_output = tgt_batch[:, 1:]  # Remove the first token for output

            # Create masks (assume `make_masks()` creates the necessary padding and lookahead masks)
            src_mask, tgt_mask = model.make_masks(src_input, tgt_input)

            optimizer.zero_grad()

            # Forward pass
            output = model(src_input, tgt_input, src_mask, tgt_mask)

            # Reshape output to match dimensions for CrossEntropyLoss
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            # Calculate loss
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

    print("Training complete!")

if __name__ == "__main__":
    # Prepare data
    data_loader = prepare_data()

    # Define the Transformer model (replace the vocab sizes with actual values)
    src_vocab_size = 10000  # You can replace this with actual vocab size
    tgt_vocab_size = 10000  # You can replace this with actual vocab size
    d_model = 512
    heads = 8
    N = 6  # Number of encoder/decoder layers
    d_ff = 2048
    dropout = 0.1

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, N, heads, d_ff, dropout)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    train(model, data_loader, epochs=10, criterion=criterion, optimizer=optimizer)
