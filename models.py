import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn_model(rnn_type, vocab_size, embedding_dim=128, max_length=200):
    """
    Builds an RNN model (LSTM or GRU).
    """
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
    
    if rnn_type == 'LSTM':
        model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    elif rnn_type == 'GRU':
        model.add(layers.GRU(64, dropout=0.2, recurrent_dropout=0.2))
    else:
        raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
        
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
