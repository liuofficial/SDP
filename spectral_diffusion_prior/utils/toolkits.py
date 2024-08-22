import numpy as np
import os
import scipy as sp


class toolkits:
    # calculate frechet inception distance
    @staticmethod
    def compute_fid(ref: np.ndarray, tar: np.ndarray):
        # ref, tar: num x features
        # calculate mean and covariance statistics
        mu1, sigma1 = ref.mean(axis=0), np.cov(ref, rowvar=False)
        mu2, sigma2 = tar.mean(axis=0), np.cov(tar, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sp.linalg.sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray, channel=False):
        assert img1.ndim == 3 and img2.ndim == 3
        img_h, img_w, img_c = img1.shape
        ref = img1.reshape(-1, img_c)
        tar = img2.reshape(-1, img_c)
        msr = np.mean((ref - tar) ** 2, 0)
        if channel is False:
            max2 = np.max(ref) ** 2  # channel-wise ???
        else:
            max2 = np.max(ref, axis=0) ** 2
        psnrall = 10 * np.log10(max2 / msr)
        out_mean = np.mean(psnrall)
        return out_mean

    @staticmethod
    def psnr_fun(ref: np.ndarray, tar: np.ndarray):
        assert ref.ndim == 4 and tar.ndim == 4
        b, c, h, w = ref.shape
        ref = ref.reshape(b, c, h * w)
        tar = tar.reshape(b, c, h * w)
        msr = np.mean((ref - tar) ** 2, 2)
        max2 = np.max(ref, axis=2) ** 2
        psnrall = 10 * np.log10(max2 / msr)
        return np.mean(psnrall)

    @staticmethod
    def sam_fun(ref: np.ndarray, tar: np.ndarray):
        assert ref.ndim == 4 and tar.ndim == 4
        b, c, h, w = ref.shape
        x_norm = np.sqrt(np.sum(np.square(ref), axis=1))
        y_norm = np.sqrt(np.sum(np.square(tar), axis=1))
        xy_norm = np.multiply(x_norm, y_norm)
        xy = np.sum(np.multiply(ref, tar), axis=1)
        dist = np.mean(np.arccos(np.minimum(np.divide(xy, xy_norm + 1e-8), 1.0 - 1.0e-9)))
        dist = np.multiply(180.0 / np.pi, dist)
        return dist

    @staticmethod
    def compute_sam(label: np.ndarray, output: np.ndarray):
        h, w, c = label.shape
        x_norm = np.sqrt(np.sum(np.square(label), axis=-1))
        y_norm = np.sqrt(np.sum(np.square(output), axis=-1))
        xy_norm = np.multiply(x_norm, y_norm)
        xy = np.sum(np.multiply(label, output), axis=-1)
        dist = np.mean(np.arccos(np.minimum(np.divide(xy, xy_norm + 1e-8), 1.0 - 1.0e-9)))
        dist = np.multiply(180.0 / np.pi, dist)
        return dist

    @staticmethod
    def check_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def channel_last(input_tensor: np.ndarray, squeeze=True):
        if squeeze is True:
            input_tensor = np.squeeze(input_tensor)
        input_tensor = np.transpose(input_tensor, axes=(1, 2, 0))
        return input_tensor

    @staticmethod
    def channel_first(input_tensor: np.ndarray, expand=True):
        input_tensor = np.transpose(input_tensor, axes=(2, 0, 1))
        if expand is True:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor