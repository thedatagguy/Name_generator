#!/usr/bin/env python
# coding: utf-8

# In[21]:


## import necessary libraries
import torch
import torch.nn as nn
import random
import time

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# In[22]:


## Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.benchmark = True


# In[23]:


### Load and preprocess data
with open("names.txt", "r", encoding="utf-8") as f:
    names = f.read().splitlines()

names = [name.lower() for name in names]

# Add start/end token
names = ["." + name + "." for name in names]

print("Sample names:", names[:5])


# In[24]:


## Create character vocabulary and padding token
chars = ['<PAD>'] + sorted(list(set("".join(names))))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

print("Vocab size:", vocab_size)


# In[25]:


## encode and decode functions
def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return ''.join([itos[i] for i in indices if itos[i] != '<PAD>'])


# In[26]:


## Prepare training data
X, Y = [], []

for name in names:
    encoded = encode(name)
    X.append(torch.tensor(encoded[:-1], dtype=torch.long))
    Y.append(torch.tensor(encoded[1:], dtype=torch.long))

class NameDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def collate_fn(batch):
    X_batch, Y_batch = zip(*batch)

    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)     # PAD = 0
    Y_padded = pad_sequence(Y_batch, batch_first=True, padding_value=-100)  # ignore in loss

    return X_padded, Y_padded

dataset = NameDataset(X, Y)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    collate_fn=collate_fn
)


# In[27]:


## Define the model
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = CharLSTM(vocab_size).to(device)
print(model)


# In[28]:


## Define optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)


# In[29]:


## Training loop
epochs = 500  # you can increase later

start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        logits = model(X_batch)

        loss = loss_fn(
            logits.view(-1, vocab_size),
            Y_batch.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("Total training time:", time.time() - start_time)


# In[30]:


## Name generation function
def generate_name(model, max_len=20, temperature=1.0, top_k=None):
    model.eval()

    idx = stoi["."]
    result = [idx]

    for _ in range(max_len):
        x = torch.tensor(result, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)

        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # TOP-K SAMPLING
        if top_k is not None:
            values, indices = torch.topk(probs, top_k)
            probs = values / values.sum()
            idx = indices[torch.multinomial(probs, 1)].item()
        else:
            idx = torch.multinomial(probs, 1).item()

        if itos[idx] == ".":
            break

        result.append(idx)

    return decode(result[1:])


# In[31]:


# GENERATE NAMES
# ==============================
print("\nGenerated Names:\n" + "="*30)

for t in [0.7, 0.9, 1.0, 1.2]:
    print(f"\nTemperature = {t}")
    for _ in range(5):
        print(generate_name(model, temperature=t, top_k=5))


# In[55]:


# NOVELTY METRIC
train_set = set([name[1:-1] for name in names])

generated = [generate_name(model, temperature=1.0) for _ in range(100)]

new_names = [n for n in generated if n not in train_set]

print("\nNovelty Evaluation")
print("Generated:", len(generated))
print("New Names:", len(new_names))
print("Novelty %:", len(new_names) / len(generated)*100)


# In[48]:


import math

def compute_perplexity(model, loader):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            logits = model(X_batch)

            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                Y_batch.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += (Y_batch != -100).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


ppl = compute_perplexity(model, loader)
print(f"\nPerplexity: {ppl:.2f}")


# In[41]:


from collections import defaultdict

bigram_counts = defaultdict(lambda: defaultdict(int))

for name in names:
    for ch1, ch2 in zip(name, name[1:]):
        bigram_counts[ch1][ch2] += 1



# In[42]:


bigram_probs = {}

for ch1 in bigram_counts:
    total = sum(bigram_counts[ch1].values())
    bigram_probs[ch1] = {
        ch2: count / total for ch2, count in bigram_counts[ch1].items()
    }


# In[43]:


import random

def generate_bigram():
    ch = "."
    name = ""

    while True:
        probs = bigram_probs[ch]
        chars = list(probs.keys())
        weights = list(probs.values())

        ch = random.choices(chars, weights=weights)[0]

        if ch == ".":
            break

        name += ch

    return name


# In[44]:


print("\nBigram Samples:")
for _ in range(10):
    print(generate_bigram())


# In[ ]:




