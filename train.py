import torch 
from torch.utils.data import Dataset, DataLoader 
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.model_selection import train_test_split 

def create_balanced_dataset(df, n_samples=20491): 
    """ 
    Create a balanced dataset with specified number of samples 
    """ 
    # Convert sentiment to numeric 
    sentiment_map = { 
        'negative': 0, 
        'neutral': 1, 
        'positive': 2 
    } 
    df['label_sentiment'] = df['sentiment'].map(sentiment_map) 
     
    # Calculate samples per class 
    samples_per_class = n_samples // 3 
     
    # Get balanced data for each class 
    balanced_dfs = [] 
    for label in range(3): 
        class_df = df[df['label_sentiment'] == label] 
        if len(class_df) > samples_per_class: 
            balanced_dfs.append(class_df.sample(n=samples_per_class, 
random_state=42)) 
        else: 
            # If we don't have enough samples, oversample 
            balanced_dfs.append(class_df.sample(n=samples_per_class, 
replace=True, random_state=42)) 
     
    # Combine balanced datasets 
    balanced_df = pd.concat(balanced_dfs) 
     
    # Shuffle the final dataset 
    return balanced_df.sample(frac=1, random_state=42)

class SentimentDataset(Dataset): 
    def __init__(self, cleaned_data, tokenizer, max_len=128): 
        self.texts = cleaned_data['Review'] 
        self.labels = cleaned_data['label_sentiment'] 
        self.tokenizer = tokenizer 
        self.max_len = max_len 
 
    def __len__(self): 
        return len(self.texts) 
 
    def __getitem__(self, idx): 
        text = str(self.texts[idx]) 
        label = self.labels[idx] 
 
        encoding = self.tokenizer.encode_plus( 
            text, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            return_token_type_ids=False, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt', 
        ) 
 
        return { 
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten(), 
            'labels': torch.tensor(label, dtype=torch.long) 
        }
    
def create_dataloader(cleaned_data, tokenizer, batch_size, max_len=128, shuffle=True):
    dataset = SentimentDataset(cleaned_data, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)    