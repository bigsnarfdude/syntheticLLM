import argparse
import json
import pandas as pd
from pipelines.cot_generation import CoTGenerator, CoTConfig
from pipelines.reward_annotation import RewardCalculator
from pipelines.augmentation import AugmentationEngine
from utils.data_utils import load_data, save_data

def parse_jsonl(file_path):
    """
    Parse JSONL file and convert to DataFrame
    
    Args:
        file_path (str): Path to JSONL file
    
    Returns:
        pd.DataFrame: Parsed conversation data
    """
    conversations = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            message = json.loads(line.strip())
            
            # Group messages by message_tree_id
            tree_id = message.get('message_tree_id')
            if tree_id not in conversations:
                conversations[tree_id] = {
                    'ID': tree_id,
                    'created_date': message.get('created_date'),
                    'user': '',
                    'assistant': '',
                    'labels': {},
                    'metadata': {}
                }
            
            # Populate conversation details
            if message.get('role') == 'prompter':
                conversations[tree_id]['user'] = message.get('text', '')
                # Capture user-specific metadata
                conversations[tree_id]['labels'] = message.get('labels', {})
                conversations[tree_id]['metadata'] = {
                    key: value for key, value in message.items() 
                    if key not in ['text', 'role', 'message_tree_id']
                }
            elif message.get('role') == 'assistant':
                conversations[tree_id]['assistant'] = message.get('text', '')
    
    # Convert to DataFrame, filtering out incomplete conversations
    df = pd.DataFrame.from_dict(conversations, orient='index')
    df = df[df['user'] != '']  # Remove entries without user messages
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Process conversations through synthetic data generation pipeline")
    parser.add_argument('--input', type=str, required=True, 
                        help="Input JSONL file path")
    parser.add_argument('--output', type=str, required=True, 
                        help="Output file path")
    parser.add_argument('--steps', nargs='+', 
                        choices=['cot', 'rewards', 'augment'],
                        default=['cot', 'rewards', 'augment'],
                        help="Processing steps to apply")
    
    args = parser.parse_args()

    # Load data from JSONL
    df = parse_jsonl(args.input)
    
    # Process pipeline steps
    if 'cot' in args.steps:
        cot_engine = CoTGenerator()
        df = cot_engine.process_batch(df)
    
    if 'rewards' in args.steps:
        reward_engine = RewardCalculator()
        df = reward_engine.process_batch(df)
    
    if 'augment' in args.steps:
        augment_engine = AugmentationEngine()
        df = augment_engine.process_batch(df)
    
    # Save results
    save_data(df, args.output)

if __name__ == "__main__":
    main()
