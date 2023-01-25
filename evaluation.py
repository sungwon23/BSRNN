#!/usr/bin/env python
# coding: utf-8

import numpy as np
from module import *
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import argparse
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf

@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=512, hop=128, save_tracks=False):
    name = os.path.split(audio_path)[-1]
    noisy, sr = librosa.load(audio_path,sr=16000)
    noisy_pad = np.pad(noisy,hop, mode='reflect')
    noisy_pad = torch.Tensor(noisy_pad).unsqueeze(0).cuda()
    assert sr == 16000
    
    length = len(noisy)
    
    noisy_spec = torch.stft(noisy_pad, n_fft, hop, window=torch.hann_window(n_fft).cuda(),return_complex=True)
    est_spec = model(noisy_spec)
        
    est_audio = torch.istft(est_spec, n_fft, hop, window=torch.hann_window(n_fft).cuda())
    est_audio = torch.flatten(est_audio[:,hop:length+hop]).cpu().numpy()
    
    assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir):
    n_fft = 512
    model = BSRNN(num_channel=64, num_layer=5).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    for audio in tqdm(audio_list):
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        est_audio, length = enhance_one_track(model, noisy_path, saved_dir, 16000*2, n_fft, n_fft//4, save_tracks)
        noisy_audio, sr = librosa.load(noisy_path,sr=16000)
        clean_audio, sr = librosa.load(clean_path,sr=16000)
        assert sr == 16000        
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics
    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 'covl: ',
          metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./saved_model/checkpoint',
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='../../dataset/VCTK-DEMAND/test/',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

args, _ = parser.parse_known_args()


if __name__ == '__main__':
    noisy_dir = os.path.join(args.test_dir, 'noisy')
    clean_dir = os.path.join(args.test_dir, 'clean')
    evaluation(args.model_path, noisy_dir, clean_dir, args.save_tracks, args.save_dir)

