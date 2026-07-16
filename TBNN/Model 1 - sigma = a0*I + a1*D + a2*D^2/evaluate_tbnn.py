import torch
import numpy as np
import matplotlib.pyplot as plt
data = torch.load(
    "tbnn_representation_theorem.pt",
    map_location="cpu",
    weights_only=False
)

print(data.keys())
