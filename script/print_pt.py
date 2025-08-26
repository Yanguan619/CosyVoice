import torch
from torch import nn

model: nn.Module = torch.load("async_cosyvoice/CosyVoice2-0.5B/spk2info.pt")

print(model.keys())
