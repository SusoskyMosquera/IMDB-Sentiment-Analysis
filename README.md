# IMDB Sentiment Analysis Project

This project implements a sentiment analysis system for movie reviews using Recurrent Neural Networks (RNNs), specifically LSTM and GRU architectures. Additionally, it provides an interactive web interface developed with FastAPI.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Data:**
    Ensure the `aclImdb` folder is in the root directory. It must contain the `train` and `test` folders, as well as the `imdb.vocab` file.

## Usage

To run the web interface, use the following command:

```bash
python app.py
```

## Model Re-evaluation

To retrain the models and evaluate them again, follow these steps:

### 1\. Train the Models

Train both models (LSTM and GRU), compare their performance, and save the best one.

```bash
python train.py
```

  * It will generate the `best_model.keras` file and training history plots.
  * It will save the comparative results in `model_comparison.csv`.

### 2\. Evaluate the Best Model

Run a detailed evaluation on the test set (Accuracy, Precision, Recall, F1, Confusion Matrix).

```bash
python evaluate.py
```

  * It will print the metrics to the console and save the confusion matrix as `confusion_matrix.png`.
  * It will show examples of correct and incorrect predictions.

### 3\. Run the Web Interface

Start the FastAPI server to use the model interactively.

```bash
python app.py
```

  * Open your browser and go to `http://127.0.0.1:8000`.
  * Enter a movie review (in English) to see the sentiment prediction.

## Project Structure

  * `data_loader.py`: Utilities for loading text data and creating the vectorization layer.
  * `models.py`: Definition of neural network architectures (LSTM and GRU).
  * `train.py`: Main script to train the models and save the best one.
  * `evaluate.py`: Script to evaluate the saved model and analyze errors.
  * `app.py`: Web application with FastAPI.
  * `requirements.txt`: List of Python dependencies.
