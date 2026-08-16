import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Tokenizer class
- Browse through all sentences, pick out unique words and assign each new ID to build a vocab dict
"""

class Tokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>":0,  # padding
            "<UNK>":1   # unknown words
        }

        self.current_id = 2  # the next unique word found will be ID 2

    def build_vocab(self, sentences):
        for sentence in sentences:
            word_list = sentence.split()
            for word in word_list:
                if word not in self.vocab:
                    self.vocab[word] = self.current_id
                    self.current_id+=1

    # turn the unique tokens into batches of list of token ids
    # append 0 if the original token didn't hit the max_length
    # append 1 if the input word is unknown (not found in vocab dict)
    def encode(self, sentence, max_length):
        word_list = sentence.split()
        token_ids = []
        for word in word_list:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(1)
        
        while len(token_ids) < max_length:
            token_ids.append(0)
        
        token_ids = token_ids[:max_length]
        return token_ids
    
"""
Token Embedding & Positional Encoding

Token Embedding
- Create a lookup table (embedding) that maps a word to a vector of random numbers
- The vector size (how many numbers in the vector) is decided by the dimension, d
- Words with similar "meanings" have similar/closer values in math

Positional Encoding
- To create a "tag" for the transformer (TF) to understand the sequence and position of each word in its sentence
- This is to prevent TF from mis-handling the word sequence as it reads a lot of words at the same time
- It might think "John ate apple" and "Apple ate John" are the same thing without the positional tag

- To achieve this, we use Sine and Cosine sinusoids
- This keeps every position between -1.0 to 1.0 (to prevent explosive numbers)
- and keeps every position unique by mixing different frequencies of Sine and Cosine waves across the dimensions of words
For even positions, we use PE(pos, 2i) = sin(pos/10000 ** (2i/d_model))
For odd positions, we use PE(pos, 2i+1) = cos(pos/10000 ** (2i/d_model))

The final outcome of this stage will produce the meaning of the word + position tag
"""

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model=64, max_length=50, dropout_rate=0.3):
        super().__init__()  # required by PyTorch

        # Creating the lookup table for embedding
        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        # randomly zeroing out some elements to prevent overfitting
        # this helps the model to learn independently during training
        # Note: The dropout probability is hardcoded to 0.3 here
        self.dropout = nn.Dropout(p=dropout_rate)    
        
        # Creating a blank grid for positional waves for the positional encoding
        self.pos_encoding = torch.zeros(max_length,d_model)

        # fill the pos_encoding matrix using the paper's formulas
        # in this step, we create a 2D matrix with the same size of the embedding layer to make it simple for the value addition
        # and for tokens in 512 (example) dimensions, there will be 256 pairs (0,1 is a pair that share the same frequency/denominator)
        # and for the denominator 10000 ** (2i/d_model), it is incredibly inefficient to do pow(10000,(2i/d_model))
        # hence we convert it -> 10000 ** (2i/d_model) = e ** ((2i/d_model) * ln(10000)) to calculate the denominator for each index pair
        # x ** y = e ** (y * ln(x))
        position = torch.arange(0,max_length).unsqueeze(1).float()  # create a 1D sequence and convert it to 2D matrix
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pos_encoding[:,0::2] = torch.sin(position*denominator)
        self.pos_encoding[:,1::2] = torch.cos(position*denominator)
        
        # PyTorch expects data to come in 3D shape [batch size, sentence length, dimension]
        # we use unsqueeze to change the 2D PE matrix into [1 * max_length * d_model] shape
        self.pos_encoding = self.pos_encoding.unsqueeze(0)
        self.register_buffer('pe', self.pos_encoding)

    def forward(self, token_ids):
        # we take the ID list and pull the corresponding vector of numbers for each word (row) from the token embedding table
        word_vectors = self.token_embed(token_ids)

        # we add the positional encoding value to the word vectors
        # and because sentences might be shorter than max_length, we slice the positional encoding to match
        # token_ids.size(1) asks for the length of the input sentence, n, and we slice just the 0 to n rows with meanings
        final_vectors = word_vectors + self.pe[:, :token_ids.size(1)] 
        return self.dropout(final_vectors)
    

"""
Scaled Dot-Product Attention - Self Attention Processing

Q (Query)  : What a word is looking for (the search bar input).
K (Key)    : What other words contain (the title of the articles).
V (Value)  : The actual meaning (the text inside the articles).

Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V

"""

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q,K,V):
        # d_k is the dimension size of the Key vector. We need it for scaling.
        # .size() tells us the shape. Because this is called from MultiHeadAttention, the tensors are 4D.
        # e.g., Q -> [1, 8, 10, 64] (1 batch, 8 heads, 10 words, 64 dimensions per word)
        # Q.size(-1) returns the last dimension -> 64, which we use for the scaling factor
        d_k = Q.size(-1) 

        # Multiply Q and K-transposed (so the rows and columns align to allow matrix multiplication)
        # ori_K -> [1, 8, 10, 64], K (after transpose) -> [1, 8, 64, 10]
        # K [1, 8, 64, 10] is multiplied by Q [1, 8, 10, 64]
        K = K.transpose(-2,-1)
        scores = torch.matmul(Q,K) # calculate the similarity between each word
        # we get a [1, 8, 10, 10] size grid of a scoreboard per head (assuming a 10-word sequence)

        # Scale the scores to prevent the numbers from getting too huge
        # when we multiply vectors with 64 dimensions, the numbers get too huge
        # when NN tries to process large numbers, the gradients will be close to 0 (vanish), causing it to stop learning
        scaled_scores = scores/math.sqrt(d_k)

        # Apply Softmax to turn the scores into percentages (0.0 to 1.0)
        # the scaled_scores are just raw numbers like 4.3, -3.5,..., and we use softmax to convert them into percentages
        # we use dim=-1 (the last dimension) to make sure each word's percentages add up to 100% across the sentence length
        attention_weights = F.softmax(scaled_scores, dim=-1)

        # Multiply the attention percentages by the Value (V) matrix - "How heavily affected (score) and how it is affected (value)"
        final_output = torch.matmul(attention_weights,V)

        return final_output, attention_weights
    

"""
Multi-Head Attention
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = d_model // num_heads    # calculate how big each head will be 
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)
        
        self.attention = ScaledDotProductAttention()

    def forward(self, x):
        """
        x shape: [batch_size, sequence_length, d_model] (e.g., [1, 10, 64])
        1 sentence, 10 words, 64 dimensions
        """
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Pass the input 'x' through the W_q, W_k, and W_v linear layers.
        # This gives us our starting Q, K, and V matrices (shape: [1, 10, 64])
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape and transpose Q, K, and V.
        # we turn [1, 10, 64] to [1, 10, 8, 8] where 512 = 8*64
        # this means the 0 to 7 dimension will be under chunk 1, 8-15 in chunk 2.. and so on
        # this is the "multi-head" part we're talking about
        # we transpose(1,2) the shape from [1, 10, 8, 8] to [1, 8, 10, 8] 
        # so PyTorch's matmul can isolate the heads and operate on the last two numbers (10 words, 64 dimensions)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)

        # Pass Q, K, and V into attention formula
        attention_output, _ = self.attention(Q, K, V)
        # (The attention_output shape is now [1, 8, 10, 8])

        
        # transpose(1,2) to turn the dimensions back to [1, 10, 8, 8]
        # .view(...,self.d_model) merges the 8 and 8 back into 64
        # contiguous ensures the transposed output is in contiguous memory for .view() to reshape the grid
        concat_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

        # Pass the concatenated output through the final W_o linear layer.
        final_output = self.W_o(concat_output)

        return final_output        
    

"""
Transformer Block
"""

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=8, ff_hidden_dim=128, ffn_dropout=0.3):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer Normalization to prevent the numbers exploding
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network (FFN) 
        # FFN is a standard linear layer with a ReLU activation function
        # ReLU - Rectified Linear Unit will show inputs directly if positive or 0 if otherwise
        self.ffn = nn.Sequential(
            nn.Linear(d_model,ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(ff_hidden_dim,d_model)
        )
    
    def forward(self, x):
        # Attention Phase with Residual Connection
        # this part ensures the original word 'meaning' is not forgotten by the bot
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # Feed-Forward Phase with another Residual Connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
    
"""
Final Chatbot Model
"""

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, num_intents, d_model=64, max_length=50, ff_hidden_dim=128,
                 num_heads=8, num_blocks=2, embedder_dropout=0.3, ffn_dropout=0.3, classifier_dropout=0.3):
        super().__init__()

        # MultiHeadAttention splits d_model evenly across num_heads (head_dim = d_model // num_heads).
        # If it doesn't divide evenly, some dimensions silently get dropped by the integer
        # division instead of raising an error — this check catches that at construction time
        # instead of letting it fail quietly deep inside the attention math.
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) "
            f"so every head gets an equal, whole share of the embedding dimensions."
        )

        self.embedding = Embedder(vocab_size, d_model, max_length, dropout_rate=embedder_dropout)
        self.dropout = nn.Dropout(p=classifier_dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                ff_hidden_dim=ff_hidden_dim,
                ffn_dropout=ffn_dropout
            ) for _ in range(num_blocks)
        ])

        self.classifier = nn.Linear(d_model, num_intents)

    def forward(self, token_ids):
        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x=block(x)
        
        # x is currently [batch, seq_length, d_model] (e.g., [batch, 10 or 50 words, 512 dims]). 
        # We average the sequence of words together to get ONE overall meaning for the whole sentence.
        # shape becomes [batch, 512]
        x = x.mean(dim=1) 
        
        # Make a prediction
        # shape becomes [batch, num_intents]
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits