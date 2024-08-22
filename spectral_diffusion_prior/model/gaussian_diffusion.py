"""
2023/03/05, uncondition
"""
import torch
import torch.nn as nn
import torch.nn.functional as func

import math
import numpy as np

from tqdm.auto import tqdm


# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# some helpful functions to compute loss
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +
                  ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image, using the eq(13) of the paper DDPM.
    """
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(x < -0.999, log_cdf_plus,
                            torch.where(x > 0.999,
                                        log_one_minus_cdf_min,
                                        torch.log(cdf_delta.clamp(min=1e-12))))

    return log_probs


class GaussianDiffusion:
    def __init__(
            self,
            denoise_fn,
            timesteps=1000,
            beta_schedule='linear',
            improved=False
    ):
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        self.improved = improved  # improved DDPM

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = func.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        # self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, x_t, t, clip_denoised=True):
        if self.improved is False:
            # predict noise using model
            pred_noise = self.denoise_fn(x_t, t)
            # get the predicted x_0: different from the algorithm2 in the paper
            x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, min=-1., max=1.)
            model_mean, model_variance, model_log_variance = \
                self.q_posterior_mean_variance(x_recon, x_t, t)
        else:
            # predict noise and variance_vector using model
            model_output = self.denoise_fn(x_t, t)
            pred_noise, pred_variance_v = torch.chunk(model_output, 2, dim=1)
            # compute predicted variance by eq(15) in the paper
            min_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)  # beta_t~
            max_log_variance = self._extract(torch.log(self.betas), t, x_t.shape)  # beta_t
            # The predict value is in [-1, 1], we should convert it to [0, 1]
            frac = (pred_variance_v + 1.) / 2.
            model_log_variance = frac * max_log_variance + (1. - frac) * min_log_variance
            model_variance = torch.exp(model_log_variance)
            # get the predicted x_0: different from the algorithm2 in the paper
            x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, min=-1., max=1.)
            model_mean, _, _ = \
                self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, model_variance, model_log_variance

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, x_t, t, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, shape, device, continuous=False, idx=None):
        batch_size = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        if continuous is False:
            for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
                img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            return img
        else:
            imgs = [img.cpu().detach().numpy()]
            for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
                img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long))
                if idx[i] == 1:
                    imgs.append(img.cpu().detach().numpy())
            return imgs

    # sample new images
    @torch.no_grad()
    def sample(self, shape, device, continuous=False, idx=None):
        return self.p_sample_loop(shape=shape, device=device, continuous=continuous, idx=idx)

    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
            self,
            x_cond,
            image_size,
            batch_size=8,
            channels=3,
            ddim_timesteps=50,
            ddim_discr_method="uniform",
            ddim_eta=0.0,
            clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = x_cond.device
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)

            # 2. predict noise using model
            pred_noise = self.denoise_fn(sample_img, x_cond, t)

            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(
                sample_img)

            sample_img = x_prev

        return sample_img

    # compute train losses
    def train_losses(self, x_start):
        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device).long()
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.denoise_fn(x_noisy, t)
        loss = func.mse_loss(noise, predicted_noise)
        # x_recon = self.predict_start_from_noise(x_noisy, t, predicted_noise)
        return loss

    # use fast sample of DDPM+
    @torch.no_grad()
    def fast_sample(
            self,
            model,
            image_size,
            batch_size=8,
            channels=3,
            timestep_respacing="50",
            clip_denoised=True):
        # make timestep sequence
        # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/respace.py
        section_counts = [int(x) for x in timestep_respacing.split(",")]
        size_per = self.timesteps // len(section_counts)
        extra = self.timesteps % len(section_counts)
        start_idx = 0
        timestep_seq = []
        for i, section_count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"cannot divide section of {size} steps into {section_count}"
                )
            if section_count <= 1:
                frac_stride = 1
            else:
                frac_stride = (size - 1) / (section_count - 1)
            cur_idx = 0.0
            taken_steps = []
            for _ in range(section_count):
                taken_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            timestep_seq += taken_steps
            start_idx += size
        total_timesteps = len(timestep_seq)
        # previous sequence
        timestep_prev_seq = np.append(np.array([-1]), timestep_seq[:-1])

        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        for i in tqdm(reversed(range(0, total_timesteps)), desc='sampling loop time step', total=total_timesteps):
            t = torch.full((batch_size,), timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), timestep_prev_seq[i], device=device, dtype=torch.long)

            # get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod_prev, prev_t + 1, sample_img.shape)

            # predict noise using model
            model_output = model(sample_img, t)
            pred_noise, pred_variance_v = torch.chunk(model_output, 2, dim=1)
            # compute beta_t and beta_t~ by eq(19) in the paper
            new_beta_t = 1. - alpha_cumprod_t / alpha_cumprod_t_prev
            new_beta_t2 = new_beta_t * (1. - alpha_cumprod_t_prev) / (1. - alpha_cumprod_t)
            min_log_variance = torch.log(new_beta_t2)  # beta_t~
            max_log_variance = torch.log(new_beta_t)  # beta_t
            # compute predicted variance by eq(15) in the paper
            # The predict value is in [-1, 1], we should convert it to [0, 1]
            frac = (pred_variance_v + 1.) / 2.
            model_log_variance = frac * max_log_variance + (1. - frac) * min_log_variance

            # get the predicted x_0: different from the algorithm2 in the paper
            x_recon = self.predict_start_from_noise(sample_img, t, pred_noise)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, min=-1., max=1.)
            mean_coef1 = (new_beta_t * torch.sqrt(alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t))
            mean_coef2 = ((1.0 - alpha_cumprod_t_prev) * torch.sqrt(1.0 - new_beta_t) / (1.0 - alpha_cumprod_t))
            model_mean = mean_coef1 * x_recon + mean_coef2 * sample_img

            noise = torch.randn_like(sample_img)
            # no noise when t == 0
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(sample_img.shape) - 1))))
            # compute x_{t-1}
            x_prev = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

            sample_img = x_prev

        return sample_img.cpu().numpy()

    # compute train losses
    def train_ddpm_plus_losses(self, model, x_start, t):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)

        # predict
        model_output = model(x_noisy, t)
        pred_noise, pred_variance_v = torch.chunk(model_output, 2, dim=1)

        # compute VLB loss
        # only learn variance, but use frozen predicted noise
        frozen_output = torch.cat([pred_noise.detach(), pred_variance_v], dim=1)
        # ground truth
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start, x_noisy, t)
        # prediction
        model_mean, _, model_log_variance = self.p_mean_variance(
            lambda *args, r=frozen_output: r,  # use a simple lambda
            t,
            clip_denoised=False
        )
        # for t > 0, compute KL
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = torch.mean(kl, dim=[1, 2, 3]) / np.log(2.0)  # use 2 for log base
        # for t = 0, compute NLL
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, model_mean, 0.5 * model_log_variance)
        decoder_nll = torch.mean(decoder_nll, dim=[1, 2, 3]) / np.log(2.0)
        vlb_loss = torch.where((t == 0), decoder_nll, kl)
        # reweight VLB
        vlb_loss *= self.timesteps / 1000

        # compute MSE loss
        mse_loss = torch.mean((pred_noise - noise) ** 2, dim=[1, 2, 3])

        loss = (mse_loss + vlb_loss).mean()
        return loss
