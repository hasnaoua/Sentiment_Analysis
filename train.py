import torch 
import pandas as pd
from torch.utils.data import Dataset, DataLoader 
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.model_selection import train_test_split 

from preprocess_text import *


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Debugging print to check the contents of texts and labels
        print(f"Sample text: {self.texts[0]}")  # Debug print to inspect the first text
        print(f"Sample label: {self.labels[0]}")  # Debug print to inspect the first label

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # Tokenization of text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
def create_dataloader(cleaned_data, tokenizer, batch_size, max_len=128, shuffle=True):
    dataset = SentimentDataset(cleaned_data, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)    


def train_sentiment_model(df, test_size=0.2, num_epochs=3, batch_size=16, learning_rate=2e-5):
    # Create a balanced dataset
    print("Creating balanced dataset...")
    balanced_df = create_balanced_dataset(df, n_samples=5000)
    print(f"Columns after balancing: {balanced_df.columns}")  # Debugging line
    print(f"Class distribution:\n{balanced_df['label_sentiment'].value_counts()}")

    # Split the data
    train_df, test_df = train_test_split(
        balanced_df, test_size=test_size, random_state=42, stratify=balanced_df['label_sentiment']
    )

    # Ensure 'processed_text' exists in both splits
    print(f"Columns in train_df: {train_df.columns}")  # Debugging line
    print(f"Columns in test_df: {test_df.columns}")    # Debugging line

    # Get texts and labels
    try:
        train_texts, train_labels = train_df['processed_text'].tolist(), train_df['label_sentiment'].tolist()
        test_texts, test_labels = test_df['processed_text'].tolist(), test_df['label_sentiment'].tolist()
    except KeyError as e:
        print(f"KeyError: {e}")
        return

    # Debugging: Print first few samples from train_texts and train_labels
    print(f"Sample of train texts: {train_texts[:3]}")
    print(f"Sample of train labels: {train_labels[:3]}")

    # Initialize tokenizer
    print("Initializing RoBERTa tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print("Initializing RoBERTa model...")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"Training on {device}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        test_preds, test_true = [], []

        print("\nEvaluating...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                test_preds.extend(preds.cpu().numpy())
                test_true.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(test_true, test_preds)
        print(f'Average training loss: {total_loss / len(train_loader):.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')

        # Detailed classification report
        print('\nClassification Report:')
        print(classification_report(test_true, test_preds, target_names=['negative', 'neutral', 'positive']))

    # Save the model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer': tokenizer
    }, 'roberta_sentiment_model_balanced.pth')

    return model, tokenizer

def predict_sentiment(text, model, tokenizer, device):
    # Set model to evaluation mode
    model.eval()

    # Tokenize and encode the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Move tensors to the specified device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make predictions without computing gradients
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).cpu().item()
        
        # Calculate probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

    # Define sentiment mapping
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Return the predicted sentiment and confidence score
    return sentiment_map[pred], probs[pred]

if __name__ == "__main__":
    # Assuming df is your input dataframe
    print("Starting training process...")

    data = pd.read_csv('./data/tripadvisor_hotel_reviews.csv', encoding="utf-8", encoding_errors="replace")
    cleaned_data = preprocess_data(data)
    cleaned_data = tokenize_text(cleaned_data)
    cleaned_data = remove_stopwords(cleaned_data)
    cleaned_data = lemmatize_text(cleaned_data)
    cleaned_data = join_tokens(cleaned_data)
    tfidf_matrix, tfidf_vectorizer = create_tfidf(cleaned_data)

    # Train the sentiment model
    model, tokenizer = train_sentiment_model(cleaned_data)

    # Example text for prediction
    example_text = "The geosolutions technology will leverage benefon GPS solutions"

    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Predict sentiment for the example text
    sentiment, confidence = predict_sentiment(example_text, model, tokenizer, device)

    # Output the prediction result
    print(f"\nExample prediction for: '{example_text}'")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.2f})")