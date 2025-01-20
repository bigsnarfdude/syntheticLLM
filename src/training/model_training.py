import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ConversationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe['input'].tolist()
        self.intents = dataframe['intent'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(self.intents[idx], dtype=torch.long)
        }

class IntentClassificationTrainer:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)

    def prepare_data(self, train_path, val_path):
        # Load data
        train_df = pd.read_json(train_path, lines=True)
        val_df = pd.read_json(val_path, lines=True)

        # Encode labels
        le = LabelEncoder()
        train_df['intent'] = le.fit_transform(train_df['intent'])
        val_df['intent'] = le.transform(val_df['intent'])

        # Create datasets
        train_dataset = ConversationDataset(train_df, self.tokenizer)
        val_dataset = ConversationDataset(val_df, self.tokenizer)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        return train_loader, val_loader, le.classes_

    def train(self, train_loader, val_loader, epochs=3):
        # Prepare optimizer and schedule
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0

            for batch in train_loader:
                # Zero gradients
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    labels=batch['label'].to(self.device)
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            # Validation
            self.model.eval()
            total_val_loss = 0
            predictions, true_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device),
                        labels=batch['label'].to(self.device)
                    )

                    loss = outputs.loss
                    total_val_loss += loss.item()

                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    predictions.extend(preds)
                    true_labels.extend(batch['label'].numpy())

            # Calculate accuracy
            accuracy = np.mean(np.array(predictions) == np.array(true_labels))
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {total_train_loss/len(train_loader):.4f}')
            print(f'Val Loss: {total_val_loss/len(val_loader):.4f}')
            print(f'Accuracy: {accuracy:.4f}')

    def save_model(self, save_path='./intent_classification_model'):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

# Example usage
if __name__ == "__main__":
    trainer = IntentClassificationTrainer()
    
    # Paths to your training and validation data
    train_path = 'data/training/train/training_data.jsonl'
    val_path = 'data/training/validation/training_data.jsonl'
    
    # Prepare data
    train_loader, val_loader, intent_labels = trainer.prepare_data(train_path, val_path)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Save model
    trainer.save_model()
