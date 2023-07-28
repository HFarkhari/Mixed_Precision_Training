# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch
import numpy as np
import time
from IPython import display

from torch.cuda.amp import GradScaler #, autocast
from torch import autocast

scaler = GradScaler()
max_norm = 1.0

N = 64_000
D_in   = 10240
D_out  = 5120
epochs = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device("cuda") if \
#     torch.cuda.is_available() else torch.device("cpu")
print('Using device:', device)

x = torch.rand(N, D_in).to(device)
y = torch.rand(N, D_out).to(device)

# Creates model and optimizer in default precision
model     = torch.nn.Linear(D_in, D_out).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

st = time.time()

for t in np.arange(epochs):

    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    with autocast(device_type=device, enabled=True, dtype=torch.float16):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred.float(), y)

##### Exits the context manager before backward()
#     loss.backward()
#     optimizer.step()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
    
ed = time.time()
print(f'time: {np.round((ed-st),2)} sec')

