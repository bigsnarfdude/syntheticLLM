# syntheticLLM

A system for capturing and processing chat conversations into training data for Chain-of-Thought (CoT) and Reinforcement Learning (RL) models.

## Project Structure

```bash
syntheticLLM/
├── data/
│   ├── raw/                # Raw conversation data
│   ├── processed/          # Intermediate processed data
│   └── training/           # Final training datasets
├── pipelines/              # Data processing modules
│   ├── cot_generator.py
│   ├── reward_calculator.py
│   ├── augmentation_engine.py
│   └── enhanced_data_processor.py
└── utils/
    ├── data_utils.py
    ├── chat_writer.py
    └── training_data_export.py
```

## Key Components

| Component | Purpose |
|----------|---------|
| `chat_writer.py` | Logs raw conversations |
| `orchestrator.py` | Coordinates data processing pipeline |
| `cot_generator.py` | Generates Chain-of-Thought steps |
| `reward_calculator.py` | Calculates conversation rewards |
| `augmentation_engine.py` | Creates data augmentations |

## Setup

1. **Install Requirements**
```bash
pip install pandas torch transformers sentence-transformers
```

2. **Prepare Data Processing**
```bash
# Recommended workflow
python orchestrator.py \
  --input data/raw/conversations.jsonl \
  --output data/training/final_dataset.parquet \
  --steps cot rewards augment
```

## Detailed Processing Steps

### 1. Capture Conversations
```python
from utils.chat_writer import ConversationLogger

logger = ConversationLogger()
logger.save_conversation("User query", "Assistant response")
```

### 2. Process Conversations
```python
# Apply processing pipeline
python orchestrator.py \
  --input data/raw/conversations.jsonl \
  --output data/processed/enhanced_conversations.parquet \
  --steps cot rewards
```

### 3. Export Training Data
```python
# Generate final training dataset
python orchestrator.py \
  --input data/processed/enhanced_conversations.parquet \
  --output data/training/final_dataset.parquet \
  --steps augment
```

## Customization

- Modify pipeline steps in `orchestrator.py`
- Adjust model configurations in respective pipeline files
- Change data paths as needed

## Features

- Conversation logging
- Chain-of-Thought generation
- Reward calculation
- Data augmentation
- Training data export

## Planned Improvements

- Advanced intent classification
- Dialogue flow modeling
- Response quality analysis
