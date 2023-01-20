# Sections to train Reward Model (RM)

Trainer code based on huggingface. Compatible with deepspeed or accelerate

Requirements

```
wandb
evaluate
datasets
transformers
torch==1.12
```

Start training reward model

```bash
python trainer.py configs/flan-t5-webgpt.yml
```

The four summary are :

- overall

- accuracy

- coverage

- coherence

### Changes
- Implemented contrastive loss
- Added an MLP embedding_size => 256 => 1 on top of frozen FlanT5 encoder. Embeddings are flattened to scalars, cosine similarity is computed between token sequences.
- Delicate assumptions: 
	1. loss function is the sum over batch examples losses
	2. No FlanT5 finetuning (frozen)

## Dataset

For now we only supports webgpt and summary dataset from OpenAI. Once
open-asisstant dataset are available it will be added here.

## Model

Check out configs

```
Open-Assistant/model/reward/instructor/configs/
    bloomz-560m.yml
    electra-base-dis-webgpt.yml
    galactica-125m.yml
    galactica-1b.yml
```

You can add new huggingface model as you want.
