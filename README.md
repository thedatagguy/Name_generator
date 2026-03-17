# Name Generator RNN (PyTorch)

This project trains a character-level LSTM model to generate human-like names from a dataset in `names.txt`.

Main notebook: `Name_geneerator.ipynb`

## Project Overview

The model learns next-character prediction:
- Input: a sequence of characters from a name
- Target: the same sequence shifted by one position
- Special token `.` is used as both start and end token

Example:
- Name: `.emma.`
- Input (`X`): `.emma`
- Target (`Y`): `emma.`

## What We Did

1. Loaded and lowercased names from `names.txt`
2. Added start/end token (`.`) to each name
3. Built character vocabulary (`stoi` / `itos`)
4. Encoded names into integer sequences
5. Created a custom `Dataset` + padded `DataLoader`
6. Built a character-level LSTM model:
   - Embedding -> LSTM -> Linear layer to vocab logits
7. Trained with `CrossEntropyLoss(ignore_index=-100)` and Adam optimizer
8. Generated names using temperature-based sampling

## Bug Fixes We Applied

The notebook had a few issues that were fixed:

1. Training loss not updating:
   - `total_loss` was initialized but never incremented
   - Fixed by adding `total_loss += loss.item()` and printing average loss per epoch

2. Sampling bug during generation:
   - Sampling was done from full sequence logits, not the last timestep
   - Fixed by using `next_token_logits = logits[:, -1, :]`

3. Dataset tensor warning:
   - Code wrapped already-existing tensors with `torch.tensor(...)`
   - Fixed by cloning/detaching existing tensors

## Training Result (Latest Run)

- Device used: CPU
- Epochs: 100
- Loss trend: decreased from ~`2.56` to ~`1.63`
- Training time: ~64 seconds (machine dependent)

The model now trains successfully and generates plausible names.

## Generated Sample Names

From the latest run:
- `soreny`
- `maleena`
- `jaylie`
- `natalia`
- `mariano`
- `reyden`
- `zaniyah`

## Installation

Use Python 3.12+ (recommended from this run environment).

```bash
python -m pip install -r requirements.txt
```

## How To Run

1. Open `Name_geneerator.ipynb` in Jupyter
2. Run cells from top to bottom
3. After training, run generation cells to print sample names

Optional terminal run (if Jupyter is available in your environment):

```bash
python -m jupyter nbconvert --to notebook --execute Name_geneerator.ipynb --output Name_geneerator.ipynb
```

## Important Points / Learnings

1. Character-level models are simple but powerful for sequence generation tasks.
2. Correct timestep selection (`last token logits`) is critical in autoregressive generation.
3. Proper padding + ignored labels (`-100`) makes variable-length training stable.
4. Monitoring real average epoch loss is essential for debugging training behavior.

## Possible Improvements

1. Add train/validation split and track validation loss
2. Tune embedding size, hidden size, and learning rate
3. Add gradient clipping for extra stability
4. Save/load model checkpoints for reuse
5. Add top-k or nucleus sampling for better name diversity
