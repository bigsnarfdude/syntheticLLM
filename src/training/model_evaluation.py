import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model_path, test_data_path):
        """
        Initialize evaluator with trained model and test data
        
        Args:
            model_path (str): Path to saved model
            test_data_path (str): Path to test dataset
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Load test data
        self.test_df = pd.read_json(test_data_path, lines=True)

    def prepare_test_data(self, max_len=128):
        """
        Prepare test data for evaluation
        
        Args:
            max_len (int): Maximum sequence length
        
        Returns:
            dict: Prepared test inputs
        """
        inputs = self.tokenizer.batch_encode_plus(
            self.test_df['input'].tolist(),
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].to(self.device),
            'attention_mask': inputs['attention_mask'].to(self.device),
            'labels': torch.tensor(self.test_df['intent'].tolist()).to(self.device)
        }

    def evaluate(self):
        """
        Evaluate model performance
        
        Returns:
            dict: Evaluation metrics
        """
        # Prepare test data
        test_inputs = self.prepare_test_data()
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=test_inputs['input_ids'],
                attention_mask=test_inputs['attention_mask']
            )
            
        # Process predictions
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        true_labels = test_inputs['labels'].cpu().numpy()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1_macro': f1_score(true_labels, predictions, average='macro'),
            'classification_report': classification_report(true_labels, predictions)
        }
        
        return metrics

    def generate_confusion_matrix(self, save_path='confusion_matrix.png'):
        """
        Generate and save confusion matrix visualization
        
        Args:
            save_path (str): Path to save confusion matrix plot
        """
        test_inputs = self.prepare_test_data()
        predictions = torch.argmax(self.model(
            input_ids=test_inputs['input_ids'],
            attention_mask=test_inputs['attention_mask']
        ).logits, dim=1).cpu().numpy()
        true_labels = test_inputs['labels'].cpu().numpy()
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def detailed_analysis(self):
        """
        Perform detailed model performance analysis
        
        Returns:
            dict: Detailed performance insights
        """
        # Prepare test data
        test_inputs = self.prepare_test_data()
        
        # Get predictions with probabilities
        with torch.no_grad():
            outputs = self.model(
                input_ids=test_inputs['input_ids'],
                attention_mask=test_inputs['attention_mask']
            )
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Convert to numpy
        probs_np = probabilities.cpu().numpy()
        true_labels = test_inputs['labels'].cpu().numpy()
        
        # Analyze misclassifications
        misclassified_indices = np.where(
            np.argmax(probs_np, axis=1) != true_labels
        )[0]
        
        misclassification_details = []
        for idx in misclassified_indices[:10]:  # First 10 misclassifications
            misclassification_details.append({
                'text': self.test_df.iloc[idx]['input'],
                'true_label': true_labels[idx],
                'predicted_label': np.argmax(probs_np[idx]),
                'confidence': np.max(probs_np[idx])
            })
        
        return {
            'misclassification_details': misclassification_details
        }

# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator(
        model_path='./intent_classification_model',
        test_data_path='data/training/test/training_data.jsonl'
    )
    
    # Run full evaluation
    print("Evaluation Metrics:")
    metrics = evaluator.evaluate()
    print(metrics['classification_report'])
    
    # Generate confusion matrix
    evaluator.generate_confusion_matrix()
    
    # Detailed analysis
    print("\nMisclassification Analysis:")
    detailed_analysis = evaluator.detailed_analysis()
    for item in detailed_analysis['misclassification_details']:
        print(f"Text: {item['text']}")
        print(f"True Label: {item['true_label']}")
        print(f"Predicted Label: {item['predicted_label']}")
        print(f"Confidence: {item['confidence']:.4f}\n")
