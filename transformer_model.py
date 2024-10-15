import torch
import torch.nn as nn
import math

# Embedding Layer
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
    def forward(self, x):
        return self.embedding(x)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embed_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        scores, attn = self.attention(Q, K, V, mask)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        output = self.fc(scores)
        return output

# Feedforward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection and layer normalization
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward network with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(heads, d_model)
        self.encoder_attention = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output = self.attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Attention over encoder output
        attn_output = self.encoder_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feedforward network with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, N, heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc_out(dec_output)
        return output

    def make_masks(self, src, tgt):
        # Create padding mask for source and target sequences
        src_mask = create_padding_mask(src)
        tgt_padding_mask = create_padding_mask(tgt)
        
        # Create look-ahead mask for the target sequence
        look_ahead_mask = create_look_ahead_mask(tgt.size(1))
        tgt_mask = torch.max(tgt_padding_mask, look_ahead_mask)
        
        return src_mask, tgt_mask

# Padding Mask (prevents attention to padding tokens)
def create_padding_mask(seq):
    mask = (seq == 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask  # Mask with shape (batch_size, 1, 1, seq_len)

# Look-Ahead Mask (prevents attention to future tokens)
def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.uint8)
    return mask  # Shape (seq_len, seq_len)

