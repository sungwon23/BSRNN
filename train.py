#!/usr/bin/env python
# coding: utf-8

import os
import dataloader
import torch
import torch.nn.functional as F
import logging
from torchinfo import summary
import argparse
from natsort import natsorted
import librosa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from module import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=int(16000*2), help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='../../dataset/VCTK-DEMAND/',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.5, 0.5, 1],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args, _ = parser.parse_known_args()
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 512
        self.hop = 128
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.model = BSRNN(num_channel=64, num_layer=5).cuda()
#         summary(self.model, [(1, 257, args.cut_len//self.hop+1, 2)])
        self.discriminator = Discriminator(ndf=16).cuda()
# #         summary(self.discriminator, [(1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1),
# #                                      (1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1)])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.init_lr*0.98)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=args.init_lr*0.98)
        
    def train_step(self, batch, use_disc):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()
    
        self.optimizer.zero_grad()
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)
                
        est_spec = self.model(noisy_spec)
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** (0.3)
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** (0.3)
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** (0.3)
        
        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec,clean_spec)

        if use_disc is False:
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * gen_loss_GAN

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
        self.optimizer.step()
        
        est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                           onesided =True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None and pesq_score_n is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)            
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels.float()) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)

            discrim_loss_metric.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])
                
        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch,use_disc):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)
        
        est_spec = self.model(noisy_spec)
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** (0.3)
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** (0.3)
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** (0.3)

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        if use_disc is False:
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * gen_loss_GAN

        est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                           onesided =True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)
        if pesq_score is not None and pesq_score_n is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)            
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    def test(self,use_disc):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        for idx, batch in enumerate(tqdm(self.test_ds)):
            step = idx + 1
            loss, disc_loss = self.test_step(batch,use_disc)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = 'Generator loss: {}, Discriminator loss: {}'
        logging.info(
            template.format(gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_epoch, gamma=0.98)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=args.decay_epoch, gamma=0.98)
        for epoch in range(args.epochs):
            self.model.train()
            self.discriminator.train()

            loss_total = 0
            loss_gan = 0
            
            if epoch >= (args.epochs/2):
                use_disc = True
            else:
                use_disc = False
            
            for idx, batch in enumerate(tqdm(self.train_ds)):
                step = idx + 1
                loss, disc_loss = self.train_step(batch,use_disc)
                template = 'Epoch {}, Step {}, loss: {}, disc_loss: {}'
                
                loss_total = loss_total + loss
                loss_gan = loss_gan + disc_loss
                
                if (step % args.log_interval) == 0:
                    logging.info(template.format(epoch, step, loss_total/step, loss_gan/step))

            gen_loss, ckpt_loss = self.test(use_disc)
            path = os.path.join(args.save_model_dir, 'gene_epoch_' + str(epoch) + '_' + str(gen_loss)[:5]
                                                                                 + '_' + str(ckpt_loss)[:5])
            path_d = os.path.join(args.save_model_dir, 'disc_epoch_' + str(epoch))
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            torch.save(self.model.state_dict(), path)
            torch.save(self.discriminator.state_dict(), path_d)
            scheduler_G.step()
            scheduler_D.step()

def main():
    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 4, args.cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()

if __name__ == '__main__':
    main()

