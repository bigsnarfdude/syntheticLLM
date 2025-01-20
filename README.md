# syntheticLLM


A system for capturing chat conversations and processing them into training data for Chain-of-Thought (CoT) and Reinforcement Learning (RL) models.

## Project Structure

```bash
.
├── LICENSE                     # The license file for the project, specifying usage and distribution terms.
├── README.md                   # The main documentation file for the repository, providing an overview of the project.
├── chat_writer.py              # A script for saving chat input/output text.
├── data                        
│   ├── models                  # Pre-trained LLM for generating synthetic data.
│   ├── processed               # Processed chat datasets ready for use in sythetic data generation.
│   ├── raw                     # Raw, unprocessed chats.
│   └── training                # RL synthetic training data.
├── main.py                     # The main entry point script for running the project.
└── src                         
    └── pipeline                
        ├── COT_generator.py    # Script for generating Chain-of-Thought (COT) reasoning data.
        ├── augmentation_engine.py  # Pipeline-specific augmentation logic and generated data.
        ├── loader.py           # Script for loading datasets into the pipeline.
        └── reward_calculator.py # Script for calculating rewards, reinforcement learning and evaluation.
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

## Maintenance

- **Versioning**: Manually rename files with dates/versions
- **Cleanup**: Old files in `processed/` can be archived
- **Monitoring**: Check `data/models/ollama/` for model updates
