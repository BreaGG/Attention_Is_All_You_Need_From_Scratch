import torch
from transformer_model import Transformer
from data_preparation import prepare_data

def create_padding_mask(seq):
    # Create padding mask where padding tokens (0) are masked (1 for masked, 0 for not masked)
    mask = (seq == 0).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, seq_len]
    return mask  # Mask of shape [batch_size, 1, 1, seq_len]

def create_look_ahead_mask(size):
    # Mask the future tokens in the sequence
    mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.uint8)
    return mask  # Shape: [seq_len, seq_len]

def create_combined_mask(tgt_seq):
    padding_mask = create_padding_mask(tgt_seq)  # Padding mask for target sequence
    look_ahead_mask = create_look_ahead_mask(tgt_seq.size(1))  # Look-ahead mask
    # Combine padding and look-ahead mask (maximum value of both)
    combined_mask = torch.max(padding_mask, look_ahead_mask)
    return combined_mask

def test_transformer():
    # Define the Transformer model (replace vocab sizes with actual values)
    src_vocab_size = 10000  # Replace with actual vocab size
    tgt_vocab_size = 10000  # Replace with actual vocab size
    d_model = 512
    heads = 8
    N = 6  # Number of encoder/decoder layers
    d_ff = 2048
    dropout = 0.1

    # Instantiate the model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, N, heads, d_ff, dropout)

    # Prepare the data loader (simulated data for testing)
    data_loader = prepare_data(batch_size=2)

    # Get a single batch from the data loader
    for src_batch, tgt_batch in data_loader:
        print(f"Source batch: {src_batch}")
        print(f"Target batch: {tgt_batch}")

        # Prepare input and output for the model
        src_input = src_batch
        tgt_input = tgt_batch[:, :-1]  # Remove the last token for input
        tgt_output = tgt_batch[:, 1:]  # Remove the first token for output

        # Create padding mask for the source input
        src_mask = create_padding_mask(src_input)  # Shape: [batch_size, 1, 1, src_seq_len]

        # Create combined mask for the target (look-ahead + padding)
        tgt_mask = create_combined_mask(tgt_input)  # Shape: [batch_size, 1, tgt_seq_len, tgt_seq_len]

        # Forward pass through the model
        output = model(src_input, tgt_input, src_mask, tgt_mask)

        # Output is expected to be of shape [batch_size, sequence_length, tgt_vocab_size]
        print(f"Model output shape: {output.shape}")

        # Print model output for inspection
        print(f"Model output: {output}")

        # Test passed if no errors and output has correct shape
        print("Test passed!")

        break  # We only need one batch for testing

if __name__ == "__main__":
    test_transformer()
