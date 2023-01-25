# BSRNN

Unofficial PyTorch implementation of the paper "HIGH FIDELITY SPEECH ENHANCEMENT WITH BAND-SPLIT RNN" (https://arxiv.org/abs/2212.00406) on VCTK-DEMAND Dataset (https://datashare.ed.ac.uk/handle/10283/2791).

![image](https://user-images.githubusercontent.com/123350717/214468836-54b8c5cf-a670-4bd9-add9-f95f48a4a673.png)

## Result

Choosed parameter settings 

N (feature dimension) : 64, L (the number of lstm layers) : 5

|                   | PESQ | SSNR | STOI |
| ----------------- | ---- | ---- | ---- |
| Noisy             | 1.97 | 1.68 | 0.91 | 
| BSRNN(N=64, L=5)  | 3.10 | 9.56 | 0.95 |

Audio files are in `saved_tracks_best` folder.

## Train and inference

### 1. Dependencies:
Used packages are can be installed by:

```pip install -r requirements.txt```

### 2. Download dataset:
Download VCTK-DEMAND dataset (https://datashare.ed.ac.uk/handle/10283/2791), change the dataset dir:
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```
### 3. Train:
```
python train.py --data_dir <dir to VCTK-DEMAND dataset>
```
If you want to adjust the model parameters, change the variables in `train.py`.  
```
self.model = BSRNN(num_channel=64, num_layer=5).cuda()
```
### 4. Inference and metric evaluation:
```
python evaluation.py --test_dir <dir to VCTK-DEMAND/test> --model_path <path to the best ckpt>
```

## Reference
-  https://github.com/ruizhecao96/CMGAN (MIT License)

