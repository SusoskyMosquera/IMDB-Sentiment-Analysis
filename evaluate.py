import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_loader import load_dataset, get_vectorization_layer

# Configuration
VOCAB_FILE = os.path.join('aclImdb', 'imdb.vocab')
TEST_DIR = os.path.join('aclImdb', 'test')
MAX_TOKENS = 20000
MAX_LENGTH = 200
BATCH_SIZE = 32

def main():
    # 1. Load Data
    print("Loading test data...")
    test_texts, test_labels = load_dataset(TEST_DIR)
    test_labels = np.array(test_labels)

    # 2. Load Model
    print("Loading best model...")
    try:
        model = tf.keras.models.load_model('best_model.keras')
    except:
        print("Error: 'best_model.keras' not found. Run train.py first.")
        return

    # 3. Prepare Vectorization (Must match training)
    print("Preparing vectorization...")
    vectorize_layer = get_vectorization_layer(VOCAB_FILE, MAX_TOKENS, MAX_LENGTH)
    
    # Vectorize test data
    # We need raw texts for analysis, but vectorized for prediction
    test_ds = tf.data.Dataset.from_tensor_slices(test_texts).batch(BATCH_SIZE)
    
    def vectorize_text(text):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text)
        
    test_ds_vec = test_ds.map(vectorize_text)

    # 4. Predict
    print("Predicting...")
    predictions_prob = model.predict(test_ds_vec)
    predictions = (predictions_prob > 0.5).astype(int).flatten()

    # 5. Metrics
    acc = accuracy_score(test_labels, predictions)
    prec = precision_score(test_labels, predictions)
    rec = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)

    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 6. Analysis of Examples
    # Create a DataFrame for easier filtering
    df = pd.DataFrame({
        'text': test_texts,
        'actual': test_labels,
        'predicted': predictions,
        'prob': predictions_prob.flatten()
    })

    # Correctly classified
    correct_pos = df[(df['actual'] == 1) & (df['predicted'] == 1)].head(5)
    correct_neg = df[(df['actual'] == 0) & (df['predicted'] == 0)].head(5)

    # Incorrectly classified
    # False Negatives (Actual Pos, Pred Neg)
    false_neg = df[(df['actual'] == 1) & (df['predicted'] == 0)].head(5)
    # False Positives (Actual Neg, Pred Pos)
    false_pos = df[(df['actual'] == 0) & (df['predicted'] == 1)].head(5)

    print("\n--- Analysis Examples ---")
    
    print("\nTop 5 Correctly Classified Positive Reviews:")
    for i, row in correct_pos.iterrows():
        print(f"[{row['prob']:.4f}] {row['text'][:100]}...")

    print("\nTop 5 Correctly Classified Negative Reviews:")
    for i, row in correct_neg.iterrows():
        print(f"[{row['prob']:.4f}] {row['text'][:100]}...")

    print("\nTop 5 False Negatives (Actual Pos, Predicted Neg):")
    for i, row in false_neg.iterrows():
        print(f"[{row['prob']:.4f}] {row['text'][:100]}...")

    print("\nTop 5 False Positives (Actual Neg, Predicted Pos):")
    for i, row in false_pos.iterrows():
        print(f"[{row['prob']:.4f}] {row['text'][:100]}...")

if __name__ == "__main__":
    main()
