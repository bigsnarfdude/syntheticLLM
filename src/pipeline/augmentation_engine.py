import pandas as pd
import nlpaug.augmenter.word as naw

class AugmentationEngine:
    def __init__(self):
        self.aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', 
            action="substitute"
        )
        
    def augment_text(self, text: str, num_variants: int = 2) -> list:
        """Generate semantic-preserving variations"""
        return [self.aug.augment(text) for _ in range(num_variants)]
    
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create augmented dataset"""
        augmented = []
        for _, row in df.iterrows():
            if row['Role'] == 'user':
                variants = self.augment_text(row['Content'])
                for var in variants:
                    new_row = row.copy()
                    new_row['Content'] = var
                    new_row['is_augmented'] = True
                    augmented.append(new_row)
        return pd.concat([df, pd.DataFrame(augmented)])