# Design and Train Model here to write a story given a prompt
# TODO: Transformer like BERT and GPT

# Imports
import tensorflow as tf

def transformer_block(x, d_model, num_heads, ff_dim, dropout=0.1):
    # Multi-head self-attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model//num_heads
    )(x, x, x)
    attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Feed-forward layer
    ff_output = tf.keras.layers.Dense(ff_dim, activation='relu')(out1)
    ff_output = tf.keras.layers.Dense(d_model)(ff_output)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ff_output)
    
    return out2

# Example usage
seq_length = 64
d_model = 256
num_heads = 8
ff_dim = 512
dropout_rate = 0.1

# Create a random input tensor for testing
x = tf.random.normal((1, seq_length, d_model))

# Apply the transformer block
transformed_x = transformer_block(x, d_model, num_heads, ff_dim, dropout_rate)
print("Output shape:", transformed_x.shape)
