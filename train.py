import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_dataset, get_vectorization_layer
from models import build_rnn_model

# Configuration
VOCAB_FILE = os.path.join('aclImdb', 'imdb.vocab')
TRAIN_DIR = os.path.join('aclImdb', 'train')
TEST_DIR = os.path.join('aclImdb', 'test')
MAX_TOKENS = 20000
MAX_LENGTH = 200
EMBEDDING_DIM = 128
EPOCHS = 5
BATCH_SIZE = 32

def plot_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} - Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} - Training and Validation Loss')
    
    plt.savefig(f'{model_name}_training_plot.png')
    plt.close()

def main():
    # 1. Load Data
    print("Loading data...")
    train_texts, train_labels = load_dataset(TRAIN_DIR)
    test_texts, test_labels = load_dataset(TEST_DIR)
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # 2. Vectorization
    print("Preparing vectorization layer...")
    vectorize_layer = get_vectorization_layer(VOCAB_FILE, MAX_TOKENS, MAX_LENGTH)
    
    # Apply vectorization to texts
    # Note: For efficiency in large datasets, we usually use tf.data.Dataset.map
    # But for 25k samples, in-memory conversion is acceptable for simplicity
    print("Vectorizing data...")
    train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = train_ds.batch(BATCH_SIZE).map(vectorize_text).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).map(vectorize_text).cache().prefetch(tf.data.AUTOTUNE)

    results = []
    best_accuracy = 0
    best_model_name = ""

    # 3. Train and Evaluate Models
    for rnn_type in ['LSTM', 'GRU']:
        print(f"\nTraining {rnn_type} model...")
        model = build_rnn_model(rnn_type, MAX_TOKENS + 2, EMBEDDING_DIM, MAX_LENGTH) # +2 for OOV and padding
        
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=EPOCHS
        )
        
        plot_history(history, rnn_type)
        
        loss, accuracy = model.evaluate(test_ds)
        print(f"{rnn_type} Test Accuracy: {accuracy:.4f}")
        
        results.append({'Model': rnn_type, 'Accuracy': accuracy})
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = rnn_type
            model.save('best_model.keras')
            # Save vectorizer config or pickle it if needed for inference app
            # For simplicity, the app will recreate the layer using the vocab file

    # 4. Report Results
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)
    results_df.to_csv('model_comparison.csv', index=False)
    
    print(f"\nBest model ({best_model_name}) saved to 'best_model.keras'")

if __name__ == "__main__":
    main()
