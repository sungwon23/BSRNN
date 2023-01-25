import torch
import torch.nn as nn
from joblib import Parallel, delayed
from pesq import pesq
import numpy as np

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1.2):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score

def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score + 0.5) / 5
    return torch.FloatTensor(pesq_score).to('cuda')