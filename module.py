import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *

class BSRNN(nn.Module):
    def __init__(self, num_channel=128, num_layer=6):
        super(BSRNN, self).__init__()
        self.num_layer = num_layer
        self.band_split = BandSplit(channels=num_channel)

        for i in range(self.num_layer):
            setattr(self, 'norm_t{}'.format(i + 1), nn.GroupNorm(1,num_channel))
            setattr(self, 'lstm_t{}'.format(i + 1), nn.LSTM(num_channel,2*num_channel,batch_first=True))
            setattr(self, 'fc_t{}'.format(i + 1), nn.Linear(2*num_channel,num_channel))

        for i in range(self.num_layer):
            setattr(self, 'norm_k{}'.format(i + 1), nn.GroupNorm(1,num_channel))
            setattr(self, 'lstm_k{}'.format(i + 1), nn.LSTM(num_channel,2*num_channel,batch_first=True,bidirectional=True))
            setattr(self, 'fc_k{}'.format(i + 1), nn.Linear(4*num_channel,num_channel))

        self.mask_decoder = MaskDecoder(channels=num_channel)            
         
        # Perform initialization    
        for m in self.modules():
            if type(m) in [nn.LSTM]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            if isinstance(m, torch.nn.Linear): 
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0) 
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = torch.view_as_real(x)
        z = self.band_split(x).transpose(1,2)
        
        B, N, T, K = z.shape

        skip = z
        for i in range(self.num_layer):
            out = getattr(self, 'norm_t{}'.format(i + 1))(skip)
            out = out.transpose(1,3).reshape(B*K, T, N)            
            out, _ = getattr(self, 'lstm_t{}'.format(i + 1))(out)
            out = getattr(self, 'fc_t{}'.format(i + 1))(out)
            out = out.reshape(B, K, T, N).transpose(1,3)
            skip = skip + out
        
        for i in range(self.num_layer):
            out = getattr(self, 'norm_k{}'.format(i + 1))(skip)
            out = out.permute(0,2,3,1).contiguous().reshape(B*T, K, N)            
            out, _ = getattr(self, 'lstm_k{}'.format(i + 1))(out)
            out = getattr(self, 'fc_k{}'.format(i + 1))(out)
            out = out.reshape(B, T, K, N).permute(0,3,1,2).contiguous()
            skip = skip + out
        
        m = self.mask_decoder(skip)                
        m = torch.view_as_complex(m)
        x = torch.view_as_complex(x)
        
        s = m[:,:,1:-1,0]*x[:,:,:-2]+m[:,:,1:-1,1]*x[:,:,1:-1]+m[:,:,1:-1,2]*x[:,:,2:]
        s_f = m[:,:,0,1]*x[:,:,0]+m[:,:,0,2]*x[:,:,1]
        s_l = m[:,:,-1,0]*x[:,:,-2]+m[:,:,-1,1]*x[:,:,-1]
        s = torch.cat((s_f.unsqueeze(2),s,s_l.unsqueeze(2)),dim=2)

        return s
    
class BandSplit(nn.Module):
    def __init__(self, channels=128):
        super(BandSplit, self).__init__()
        self.band = torch.Tensor([
        2,    3,    3,    3,    3,   3,   3,    3,    3,    3,   3,
        8,    8,    8,    8,    8,   8,   8,    8,    8,    8,   8,   8,
        16,   16,   16,   16,   16,  16,  16,   17])
        for i in range(len(self.band)):
            setattr(self, 'norm{}'.format(i + 1), nn.GroupNorm(1,int(self.band[i]*2)))
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(int(self.band[i]*2),channels))

    def forward(self, x):
        hz_band = 0
        x = x.transpose(1,2)
        for i in range(len(self.band)):
            x_band = x[:,:,hz_band:hz_band+int(self.band[i]),:]
            x_band = torch.reshape(x_band,[x_band.size(0),x_band.size(1),x_band.size(2)*x_band.size(3)])
            out = getattr(self, 'norm{}'.format(i + 1))(x_band.transpose(1,2))
            out = getattr(self, 'fc{}'.format(i + 1))(out.transpose(1,2))

            if i == 0:
                z = out.unsqueeze(3)
            else: 
                z = torch.cat((z,out.unsqueeze(3)),dim=3)
            hz_band = hz_band+int(self.band[i])
        return z
    
class MaskDecoder(nn.Module):
    def __init__(self, channels=128):
        super(MaskDecoder, self).__init__()
        self.band = torch.Tensor([
        2,    3,    3,    3,    3,   3,   3,    3,    3,    3,   3,
        8,    8,    8,    8,    8,   8,   8,    8,    8,    8,   8,   8,
        16,   16,   16,   16,   16,  16,  16,   17])
        for i in range(len(self.band)):
            setattr(self, 'norm{}'.format(i + 1), nn.GroupNorm(1,channels))
            setattr(self, 'fc1{}'.format(i + 1), nn.Linear(channels,4*channels))
            setattr(self, 'tanh{}'.format(i + 1), nn.Tanh())
            setattr(self, 'fc2{}'.format(i + 1), nn.Linear(4*channels,int(self.band[i]*12)))
            setattr(self, 'glu{}'.format(i + 1), nn.GLU())

    def forward(self, x):
        for i in range(len(self.band)):
            x_band = x[:,:,:,i]
            out = getattr(self, 'norm{}'.format(i + 1))(x_band)
            out = getattr(self, 'fc1{}'.format(i + 1))(out.transpose(1,2))
            out = getattr(self, 'tanh{}'.format(i + 1))(out)
            out = getattr(self, 'fc2{}'.format(i + 1))(out)
            out = getattr(self, 'glu{}'.format(i + 1))(out)
            out = torch.reshape(out,[out.size(0),out.size(1),int(out.size(2)/6), 3, 2])
            if i == 0:
                m = out
            else:
                m = torch.cat((m,out),dim=2)
        return m.transpose(1,2)
        
class Discriminator(nn.Module):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)