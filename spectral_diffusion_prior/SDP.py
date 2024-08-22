"""
A Spectral Diffusion Prior for Unsupervised Hyperspectral Image SuperResolution, TGRS, 2024
JianJun Liu, 2024/8/21
"""
import time
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from data.data_info import DataInfo
from utils.torchkits import torchkits
from utils.toolkits import toolkits
from utils.blur_down import BlurDown
from utils.ema import EMA
from model.mlp_net import MLPSkipNetConfig, Activation
from model.gaussian_diffusion import GaussianDiffusion
from blind import Blind


# MLP for leaning spectral diffusion prior
class SpecDiffusionNet(nn.Module):
    def __init__(self, hs_bands, layers=5, timesteps=1000):
        super().__init__()
        self.hs_bands = hs_bands
        self.timesteps = timesteps

        self.net = MLPSkipNetConfig(num_channels=self.hs_bands,
                                    skip_layers=tuple(range(1, layers)),
                                    num_hid_channels=512,  # 512
                                    num_layers=layers,
                                    num_time_emb_channels=64,
                                    activation=Activation.silu,
                                    use_norm=True,
                                    condition_bias=1.0,
                                    dropout=0.001,
                                    last_act=Activation.none,
                                    num_time_layers=2,
                                    time_last_act=False).make_model()

        self.gauss_diffusion = GaussianDiffusion(denoise_fn=self.net, timesteps=self.timesteps, improved=False)

        self._init_weights()

    def forward(self, X):
        # hw, b = X.shape
        loss = self.gauss_diffusion.train_losses(X)
        return loss

    def cpt_loss(self, output, label=None):
        loss = output
        return loss

    def sample(self, batch_size, device, continuous=False, idx=None):
        shape = (batch_size, self.hs_bands)
        generated_spectrum = self.gauss_diffusion.sample(shape=shape, device=device, continuous=continuous, idx=idx)
        return generated_spectrum

    def _init_weights(self, init_type='normal'):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init_type == 'normal':
                    nn.init.xavier_normal_(m.weight.data)
                elif init_type == 'uniform':
                    nn.init.xavier_uniform_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        pass


# Class for training spectral diffusion model
class SDM(DataInfo):
    def __init__(self, ndata, nratio=8, nsnr=0):
        super().__init__(ndata, nratio, nsnr)
        lr = [1e-2, 1e-2, 1e-2, 0.5e-2]  # learning rate
        self.lr = lr[ndata]
        self.lr_fun = lambda epoch: 0.001 * max(1000 - epoch / 10, 1)
        layers = [5, 5, 5, 5]  # layers of MLP
        self.model = SpecDiffusionNet(self.hs_bands, layers=layers[ndata])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_fun)
        torchkits.get_param_num(self.model)
        toolkits.check_dir(self.model_save_path)
        for name, parameters in self.model.named_parameters():
            print(name, ':', parameters.size())
        print(self.model)
        self.model_save_pkl = self.model_save_path + 'spec.pkl'
        self.model_save_time = self.model_save_path + 't.mat'
        pass

    def convert_data(self, img):
        _, B, H, W = img.shape
        img = img.reshape(B, H * W).permute(1, 0)
        return img

    def train(self, max_iter=30000, batch_size=512):
        cudnn.benchmark = True
        fed_data = self.convert_data(torch.tensor(self.tgt)).cuda()
        model = self.model.cuda()
        model.train()
        ema = EMA(model, 0.999)
        ema.register()
        time_start = time.perf_counter()
        for epoch in range(0, max_iter):
            lr = self.optimizer.param_groups[0]['lr']
            t = np.random.randint(0, fed_data.shape[0], size=(batch_size,))
            output = self.model(fed_data[t, :])
            loss = model.cpt_loss(output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            ema.update()
            if epoch % 1000 == 0:
                tol = torchkits.to_numpy(loss)
                print(epoch, lr, tol)
                torch.save(self.model.state_dict(), self.model_save_pkl)
            if epoch == max_iter - 1:
                ema.apply_shadow()
                torch.save(ema.model.state_dict(), self.model_save_pkl)
        run_time = time.perf_counter() - time_start
        sio.savemat(self.model_save_time, {'t': run_time})

    def show(self):
        model = self.model.cuda()
        model.eval()
        model.load_state_dict(torch.load(self.model_save_pkl))
        gen_spec = model.sample(5, device='cuda', continuous=False)
        spec = torchkits.to_numpy(gen_spec)
        plt.figure(num=0)
        plt.plot(spec.T)
        plt.show()


# learning-like
class Target(nn.Module):
    def __init__(self, hs_bands, height, width):
        super(Target, self).__init__()
        self.height = height
        self.width = width
        self.img = nn.Parameter(torch.ones(1, hs_bands, height, width))
        self.img.requires_grad = True

    def get_image(self):
        # hw x b
        return self.img

    def check(self):
        self.img.data.clamp_(0.0, 1.0)


# main class for implementing spectral diffusion prior
class SDP(DataInfo):
    def __init__(self, ndata, nratio=8, nsnr=0, psf=None, srf=None):
        super().__init__(ndata, nratio, nsnr)
        self.strX = 'X.mat'
        if psf is not None:
            self.psf = psf
        if srf is not None:
            self.srf = srf
        # spec
        self.spec_net = SDM(ndata, nratio, nsnr)
        # set learning rate
        lrs = [1e-3, 1e-3, 2.5e-3, 8e-3]
        self.lr = lrs[ndata]
        self.ker_size = self.psf.shape[0]
        lams = [0.1, 0.1, 0.1, 1.0]  # parameter: lambda
        self.lam_A, self.lam_B, self.lam_C = lams[ndata], 1, 1e-6
        self.lr_fun = lambda epoch: 1.0
        # define
        self.psf = np.reshape(self.psf, newshape=(1, 1, self.ker_size, self.ker_size))
        self.psf = torch.tensor(self.psf)
        self.srf = np.reshape(self.srf, newshape=(self.ms_bands, self.hs_bands, 1, 1))
        self.srf = torch.tensor(self.srf)
        self.__hsi = torch.tensor(self.hsi)
        self.__msi = torch.tensor(self.msi)
        toolkits.check_dir(self.model_save_path)
        self.model_save_pkl = self.model_save_path + 'prior.pkl'
        self.blur_down = BlurDown()
        pass

    def cpt_loss(self, X, hsi, msi, psf, srf):
        Y = self.blur_down(X, psf, int((self.ker_size - 1) / 2), self.hs_bands, self.ratio)
        Z = func.conv2d(X, srf, None)
        loss = self.lam_A * func.mse_loss(Y, hsi, reduction='sum') + self.lam_B * func.mse_loss(Z, msi, reduction='sum')
        return loss

    def img_to_spec(self, X):
        X = X.reshape(self.hs_bands, -1)
        X = X.permute(1, 0)
        return X

    def spec_to_img(self, X):
        X = X.reshape(1, self.height, self.width, self.hs_bands)
        X = X.permute(0, 3, 1, 2)
        return X

    def train(self, gam=1e-3):
        cudnn.benchmark = True
        model = Target(self.hs_bands, self.height, self.width).cuda()
        opt = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.lam_C)
        scheduler = optim.lr_scheduler.LambdaLR(opt, self.lr_fun)
        torchkits.get_param_num(model)

        hsi = self.__hsi.cuda()
        msi = self.__msi.cuda()
        psf = self.psf.cuda()
        srf = self.srf.cuda()

        self.spec_net.model.load_state_dict(torch.load(self.spec_net.model_save_pkl))
        self.spec_net.model.to(device=msi.device)
        self.spec_net.model.eval()
        for name, param in self.spec_net.model.named_parameters():
            param.requires_grad = False

        timesteps = self.spec_net.model.timesteps
        model.train()
        ema = EMA(model, 0.9)
        ema.register()
        time_start = time.perf_counter()
        for i in range(timesteps):
            lr = opt.param_groups[0]['lr']

            # Select t
            t = timesteps - 1 - i
            t = np.array([t]).astype(int)
            t = torch.full((self.height * self.width,), t[0], device=msi.device, dtype=torch.long)

            in_steps = 3  # 3
            for j in range(in_steps):
                img = model.get_image()
                spec = self.img_to_spec(img)
                # spectral diffusion model
                noise = torch.randn_like(spec)
                xt = self.spec_net.model.gauss_diffusion.q_sample(spec, t, noise=noise)
                noise_pred = self.spec_net.model.gauss_diffusion.denoise_fn(xt, t)
                # Learned condition
                spat_spec_loss = self.cpt_loss(img, hsi, msi, psf, srf)
                spec_prior_loss = func.mse_loss(noise_pred, noise, reduction='sum')
                loss = spat_spec_loss + gam * spec_prior_loss
                # Update
                opt.zero_grad()
                loss.backward()
                opt.step()
                model.check()
                ema.update()
            pass
            scheduler.step()

            if i % 100 == 0:
                if self.ref is not None:
                    img = model.get_image()
                    psnr = toolkits.psnr_fun(self.ref, torchkits.to_numpy(img))
                    sam = toolkits.sam_fun(self.ref, torchkits.to_numpy(img))
                    print(i, psnr, sam, loss.data, lr)
            pass
        run_time = time.perf_counter() - time_start

        ema.apply_shadow()
        img = ema.model.get_image()
        # img = model.get_image()
        img = torchkits.to_numpy(img)
        psnr = toolkits.psnr_fun(self.ref, img)
        sam = toolkits.sam_fun(self.ref, img)
        print(psnr, sam)

        sio.savemat(self.save_path + self.strX, {'X': img, 't': run_time})
        pass


if __name__ == '__main__':
    ndata, nratio, nsnr = 0, 8, 0  # choose data

    # spectral diffusion model
    spec_net = SDM(ndata=ndata, nratio=nratio, nsnr=nsnr)
    spec_net.train()
    # spec_net.show()

    blind = Blind(ndata=ndata, nratio=nratio, nsnr=nsnr, blind=True)
    blind.train()
    blind.get_save_result(is_save=True)

    # spectral diffusion prior
    gams = [1e-3, 1e-3, 1e-3, 1e-1]  # parameter: gamma
    net = SDP(ndata=ndata, nratio=nratio, nsnr=nsnr, psf=blind.psf, srf=blind.srf)
    net.train(gam=gams[ndata])
