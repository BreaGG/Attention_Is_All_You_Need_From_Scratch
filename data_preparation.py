import nltk
from torch.utils.data import Dataset, DataLoader
import torch

# Download the tokenizer models
nltk.download('punkt_tab')

# Tokenizer functions using NLTK
def tokenize_en(text):
    return nltk.word_tokenize(text)

def tokenize_de(text):
    return nltk.word_tokenize(text)

# Simple Vocabulary to map words to integer indices
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def __len__(self):
        return len(self.word2idx)

    def numericalize(self, sentence):
        return [self.word2idx[word] for word in sentence]

# Dataset class using NLTK tokenization
class NLTKTranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        # Build the vocabulary for both source and target languages
        for src_text, tgt_text in data:
            self.src_vocab.add_sentence(self.src_tokenizer(src_text))
            self.tgt_vocab.add_sentence(self.tgt_tokenizer(tgt_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_tokens = self.src_tokenizer(src_text)
        tgt_tokens = self.tgt_tokenizer(tgt_text)
        src_indices = self.src_vocab.numericalize(src_tokens)
        tgt_indices = self.tgt_vocab.numericalize(tgt_tokens)
        return src_indices, tgt_indices

def prepare_data(batch_size=32):
    # Simulated dataset (you can load a real dataset here)
    data = [("Hello world", "Hallo Welt"), ("How are you?", "Wie geht es Ihnen?")]

    # Create vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    # Define dataset
    dataset = NLTKTranslationDataset(data, src_vocab, tgt_vocab, tokenize_en, tokenize_de)

    # Create DataLoader for batching
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return data_loader

# Collate function to pad sequences to the same length
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    # Find the maximum length of source and target sequences in the batch
    src_max_len = max([len(src) for src in src_batch])
    tgt_max_len = max([len(tgt) for tgt in tgt_batch])

    # Padding token (assume 0 is the padding index)
    PAD_IDX = 0

    # Pad the source and target sequences
    src_padded = [src + [PAD_IDX] * (src_max_len - len(src)) for src in src_batch]
    tgt_padded = [tgt + [PAD_IDX] * (tgt_max_len - len(tgt)) for tgt in tgt_batch]

    # Convert the lists to PyTorch tensors
    src_tensor = torch.tensor(src_padded, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long)

    return src_tensor, tgt_tensor

if __name__ == "__main__":
    data_loader = prepare_data()
    for src_batch, tgt_batch in data_loader:
        print(f"Source batch: {src_batch}")
        print(f"Target batch: {tgt_batch}")
