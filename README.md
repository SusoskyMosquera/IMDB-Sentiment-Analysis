# IMDB Sentiment Analysis Project

This project implements a sentiment analysis system for movie reviews using RNNs (LSTM/GRU) and provides a web interface using FastAPI.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data:**
    Ensure the `aclImdb` folder is in the root directory. It should contain `train` and `test` folders, and `imdb.vocab`.

## Usage

### 1. Train the Models
Train both LSTM and GRU models, compare them, and save the best one.
```bash
python train.py
```
*   This will generate `best_model.keras` and training plots.
*   It will also save `model_comparison.csv`.

### 2. Evaluate the Best Model
Run detailed evaluation on the test set (Accuracy, Precision, Recall, F1, Confusion Matrix).
```bash
python evaluate.py
```
*   This will print metrics and save `confusion_matrix.png`.
*   It will also display examples of correct and incorrect predictions.

### 3. Run the Web Interface
Start the FastAPI server to use the model interactively.
```bash
python app.py
```
*   Open your browser and go to `http://127.0.0.1:8000`.
*   Enter a movie review to see the sentiment prediction.

## Project Structure

*   `data_loader.py`: Utilities for loading text data and creating the vectorization layer.
*   `models.py`: Definition of the RNN architectures (LSTM and GRU).
*   `train.py`: Script to train models and save the best one.
*   `evaluate.py`: Script to evaluate the saved model and analyze errors.
*   `app.py`: FastAPI web application.
*   `requirements.txt`: Python dependencies.
