# syntheticLLM


A system for capturing chat conversations and processing them into training data for Chain-of-Thought (CoT) and Reinforcement Learning (RL) models.

## Project Structure

```bash
syntheticLLM/
├── data/
│   ├── models/              # Pre-trained models
│   ├── processed/
│   │   ├── with_cot/       # Chain of thought data
│   │   └── with_rewards/   # Reward-annotated data
│   ├── raw/                # Raw conversation data
│   └── training/           # Final training datasets
├── src/
│   ├── orchestrator.py
│   ├── pipeline/
│   │   ├── cot_generator.py
│   │   ├── reward_calculator.py
│   │   ├── augmentation_engine.py
│   │   └── enhanced_data_processor.py
│   └── utils/
│       ├── data_utils.py
│       └── exporters/
│           ├── cot_exporter.py
│           └── training_data_exporter.py
├── main.py
└── requirements.txt
```

## Key Components

### Data Flow
1. **Raw Data**: Collected from chat app in `data/raw/conversations.csv`
2. **Processing**: Sequential enhancement through pipeline steps
3. **Training Data**: Final datasets in `data/training/`

### Core Files
| File | Purpose |
|------|---------|
| `src/conversation_logger.py` | Collects raw chat data |
| `src/orchestrator.py` | Coordinates processing steps |
| `src/pipelines/*.py` | Individual processing modules |

## Setup

1. **Install Requirements**
```bash
pip install pandas requests sentence-transformers transformers nlpaug
```

2. **Set Up Ollama (for local CoT generation)**
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Download models
ollama pull llama2
```

## Usage

```
raw/conversations.csv 
→ processed/with_cot/... 
→ processed/with_rewards/... 
→ training/cot_training/...


# Generate CoT data
python src/orchestrator.py --input data/raw/conversations.jsonl --output data/processed/with_cot/latest.parquet --steps cot

# Calculate rewards from CoT data
python src/orchestrator.py --input data/processed/with_cot/latest.parquet --output data/processed/with_rewards/latest.parquet --steps rewards

# Full pipeline (CoT → Rewards → Augment)
python src/orchestrator.py --input data/raw/conversations.jsonl --output data/training/final.parquet --steps cot rewards augment
```

**1. Capture Conversations**
```python
from src.conversation_logger import ConversationLogger

logger = ConversationLogger()
logger.save_conversation("What's AI?", "Artificial intelligence is...")
```

**2. Process Data**
```bash
# Generate Chain-of-Thought steps
python src/orchestrator.py \
  --input data/raw/conversations.csv \
  --output data/processed/with_cot/latest.parquet \
  --steps cot

# Add RL rewards
python src/orchestrator.py \
  --input data/processed/with_cot/latest.parquet \
  --output data/processed/with_rewards/latest.parquet \
  --steps rewards
```

**3. Export Training Data**
```bash
# Create final dataset
python src/orchestrator.py \
  --input data/processed/with_rewards/latest.parquet \
  --output data/training/cot_rl_dataset.parquet \
  --steps augment
```

## Customization

1. **Change Paths**  
   Modify `orchestrator.py` input/output paths

2. **Use Different Models**  
   Edit `CoTConfig` in `pipelines/cot_generation.py`

3. **Adjust Processing**  
   Modify pipeline steps in `orchestrator.py` arguments:
   ```python
   --steps [cot|rewards|augment]
   ```

## TODO
Intent classification
CoT pattern extraction
Response quality analysis
Dialogue flow modeling

https://huggingface.co/datasets/OpenAssistant/oasst1

