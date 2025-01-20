import pandas as pd
import json
import os

class TrainingDataExporter:
    def __init__(self, output_formats=['jsonl', 'parquet', 'csv']):
        """
        Initialize data exporter with supported formats
        
        Args:
            output_formats (list): Formats to export
        """
        self.output_formats = output_formats

    def prepare_training_data(self, df):
        """
        Prepare dataframe for model training
        
        Args:
            df (pd.DataFrame): Processed dataframe
        
        Returns:
            pd.DataFrame: Training-ready dataframe
        """
        # Select and rename columns for training
        training_df = df[['text', 'predicted_intent', 'ollama_complexity']]
        training_df.columns = ['input', 'intent', 'complexity']
        
        # Add training-specific columns
        training_df['version'] = 1
        training_df['source'] = 'synthetic_data_pipeline'
        
        return training_df

    def export_data(self, df, base_path='data/training'):
        """
        Export processed data in multiple formats
        
        Args:
            df (pd.DataFrame): Processed dataframe
            base_path (str): Base directory for exports
        """
        # Ensure export directory exists
        os.makedirs(base_path, exist_ok=True)
        
        # Prepare training data
        training_df = self.prepare_training_data(df)
        
        # Export in specified formats
        for format in self.output_formats:
            export_path = os.path.join(base_path, f'training_data.{format}')
            
            if format == 'jsonl':
                training_df.to_json(export_path, orient='records', lines=True)
            elif format == 'parquet':
                training_df.to_parquet(export_path)
            elif format == 'csv':
                training_df.to_csv(export_path, index=False)
            
            print(f"Exported {format.upper()} to {export_path}")

    def split_data(self, df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Split data into train, validation, and test sets
        
        Args:
            df (pd.DataFrame): Input dataframe
            train_ratio (float): Proportion of training data
            val_ratio (float): Proportion of validation data
            test_ratio (float): Proportion of test data
        
        Returns:
            dict: Splits of the dataset
        """
        # Shuffle the dataframe
        shuffled_df = df.sample(frac=1, random_state=42)
        
        # Calculate split indices
        total_rows = len(shuffled_df)
        train_end = int(total_rows * train_ratio)
        val_end = train_end + int(total_rows * val_ratio)
        
        # Create splits
        splits = {
            'train': shuffled_df[:train_end],
            'validation': shuffled_df[train_end:val_end],
            'test': shuffled_df[val_end:]
        }
        
        return splits

# Example usage
if __name__ == "__main__":
    from ollama_enhanced_data_processor import OllamaEnhancedDataProcessor
    
    # Sample data preparation
    sample_data = pd.DataFrame({
        'text': [
            "What is the capital of France?",
            "Explain quantum computing basics.",
            "How do neural networks work?",
            "Compare machine learning algorithms."
        ]
    })
    
    # Process data
    processor = OllamaEnhancedDataProcessor()
    processed_df = processor.process_dataset(sample_data)
    
    # Export data
    exporter = TrainingDataExporter()
    
    # Export all formats
    exporter.export_data(processed_df)
    
    # Split and export data splits
    data_splits = exporter.split_data(processed_df)
    
    # Export each split
    for split_name, split_df in data_splits.items():
        exporter.export_data(split_df, f'data/training/{split_name}')
