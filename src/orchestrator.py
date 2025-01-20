import argparse
from pipelines.cot_generation import CoTGenerator, CoTConfig
from pipelines.reward_annotation import RewardCalculator
from pipelines.augmentation import AugmentationEngine
from utils.data_utils import load_data, save_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--steps', nargs='+', 
                      choices=['cot', 'rewards', 'augment'])
    args = parser.parse_args()

    # Load data
    df = load_data(args.input)
    
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