# BSRNN

Unofficial PyTorch implementation of the paper "HIGH FIDELITY SPEECH ENHANCEMENT WITH BAND-SPLIT RNN" (https://arxiv.org/abs/2212.00406) on DEMAND-Voicebank Dataset (https://datashare.ed.ac.uk/handle/10283/2791).

![image](https://user-images.githubusercontent.com/123350717/214468836-54b8c5cf-a670-4bd9-add9-f95f48a4a673.png)

## Result

Choosed parameter settings 

N (feature dimension) : 64, L (the number of lstm layers) : 5

|                   | PESQ | SSDR | STOI |
| ----------------- | ---- | ---- | ---- |
| Noisy             | 1.97 | 1.68 | 0.91 | 
| BSRNN(N=64, L=5)  | 3.10 | 9.56 | 0.95 |

Audio files are in `saved_tracks_best`

## Train and inference
