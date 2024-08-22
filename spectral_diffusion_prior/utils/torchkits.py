import numpy as np
import scipy.sparse as sp
import torch


class torchkits:
    @staticmethod
    def sparse_to_torch(input_tensor: sp.coo_matrix):
        values = input_tensor.data
        indices = np.vstack((input_tensor.row, input_tensor.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = input_tensor.shape
        input_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return input_tensor

    @staticmethod
    def extract_patches(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
        # input_tensor: N x C x H x W, patches: N * H' * W', C, h, w
        if pad_num != 0:
            input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
        all_patches = input_tensor.unfold(2, kernel, stride).unfold(3, kernel, stride)
        N, C, H, W, h, w = all_patches.shape
        all_patches = all_patches.permute(0, 2, 3, 1, 4, 5)
        all_patches = torch.reshape(all_patches, shape=(N * H * W, C, h, w))
        return all_patches

    @staticmethod
    def extract_patches_v1(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
        # input_tensor: N x C x H x W, patches: N * H' * W', C, h, w
        if pad_num != 0:
            input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
        N, C, H, W = input_tensor.shape
        unfold = torch.nn.Unfold(kernel_size=(kernel, kernel), stride=stride)
        all_patches = unfold(input_tensor)
        _, _, L = all_patches.shape
        all_patches = torch.reshape(all_patches, shape=(N, C, kernel, kernel, L))
        all_patches = all_patches.permute(0, 4, 1, 2, 3)
        all_patches = torch.reshape(all_patches, shape=(N * L, C, kernel, kernel))
        return all_patches

    @staticmethod
    def extract_patches_ex(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
        # input_tensor: N x C x H x W, patches: N * H' * W', C, h, w
        if pad_num != 0:
            input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
        all_patches = input_tensor.unfold(2, kernel, stride).unfold(3, kernel, stride)
        # N, C, H, W, h, w = all_patches.shape
        all_patches = all_patches.permute(0, 2, 3, 1, 4, 5)  # shape=(N, H, W, C, h, w)
        return all_patches

    @staticmethod
    def aggregate_patches(input_tensor: torch.Tensor, height, width, kernel, stride, pad_num=0, patch=1):
        N, C, h, w = input_tensor.shape
        dH = height + 2 * pad_num - (height + 2 * pad_num - kernel) // stride * stride - kernel
        dW = width + 2 * pad_num - (width + 2 * pad_num - kernel) // stride * stride - kernel
        height, width = height - dH, width - dW
        input_tensor = input_tensor.reshape(patch, N // patch, C, h, w)
        output_tensor = input_tensor.permute(0, 2, 3, 4, 1)
        output_tensor = torch.reshape(output_tensor, shape=(patch, C * h * w, N // patch))
        num = torch.ones_like(output_tensor)
        fold = torch.nn.Fold(output_size=(height + 2 * pad_num, width + 2 * pad_num),
                             kernel_size=(kernel, kernel),
                             stride=stride)
        output_tensor = fold(output_tensor)
        num = fold(num)
        output_tensor = output_tensor[:, :, pad_num: height + pad_num, pad_num: width + pad_num]
        num = num[:, :, pad_num: height + pad_num, pad_num: width + pad_num]
        output_tensor = output_tensor / num
        return output_tensor, dH, dW

    @staticmethod
    def torch_cb_loss(ref: torch.Tensor, tar: torch.Tensor, eps=1e-6):
        diff = ref - tar
        loss = torch.sqrt(diff * diff + eps)
        loss = torch.sum(loss)
        return loss

    @staticmethod
    def torch_norm(input_tensor: torch.Tensor, mode=1, reduce=False):
        if mode == 1:
            if reduce is False:
                loss = torch.sum(torch.abs(input_tensor))
            else:
                loss = torch.mean(torch.abs(input_tensor))
            return loss
        return None

    @staticmethod
    def torch_sam(label: torch.Tensor, output: torch.Tensor, reduce=True, angle=True):
        x_norm = torch.sqrt(torch.sum(torch.square(label), dim=-1))
        y_norm = torch.sqrt(torch.sum(torch.square(output), dim=-1))
        xy_norm = torch.multiply(x_norm, y_norm)
        xy = torch.sum(torch.multiply(label, output), dim=-1)
        dist = torch.divide(xy, torch.maximum(xy_norm, torch.tensor(1e-8)))
        dist = torch.arccos(dist)
        if angle is True:
            dist = torch.multiply(torch.tensor(180.0 / np.pi), dist)
        if reduce is True:
            dist = torch.mean(dist)
        else:
            dist = torch.sum(dist)
        return dist

    @staticmethod
    def torch_psnr(ref: torch.Tensor, tar: torch.Tensor):
        b, c, h, w = ref.shape
        ref = ref.reshape(b, c, h * w)
        tar = tar.reshape(b, c, h * w)
        msr = torch.mean(torch.pow(ref - tar, 2), dim=2)
        max2 = torch.pow(torch.max(ref, dim=2)[0], 2)
        psnrall = 10 * torch.log10(max2 / msr)
        return torch.mean(psnrall)

    @staticmethod
    def sparsity_l1_div_l2(x: torch.Tensor):
        N, C, H, W = x.shape  # perform on mode-C
        l1norm = torch.sum(torch.abs(x), dim=1)
        l2norm = torch.sqrt(torch.sum(torch.square(x), dim=1))
        sparsity = torch.sum(l1norm / l2norm)
        return sparsity

    @staticmethod
    def joint_sparsity(x: torch.Tensor):
        N, H, W = x.shape  # perform on mode H, W
        l2norm = torch.sqrt(torch.sum(torch.square(x), dim=2))
        l21norm = torch.sum(l2norm, dim=1)
        fnorm = torch.sqrt(torch.sum(torch.square(x), dim=(1, 2))) + 1e-9
        return torch.sum(l21norm / fnorm)

    @staticmethod
    def sp_joint_l1_div_l2(img: torch.Tensor, jdx: torch.Tensor):
        _, C, W, H = img.shape
        output = torch.squeeze(img)
        output = torch.reshape(output, shape=(C, W * H))
        output = torch.transpose(output, 0, 1)
        output = torch.square(output)
        output = torch.matmul(jdx, output)
        l1norm = torch.sum(torch.sqrt(output), dim=1)
        l2norm = torch.sum(output, dim=1)
        output = torch.sum(l1norm / l2norm)
        return output

    @staticmethod
    def sp_joint_l21(img: torch.Tensor, jdx: torch.Tensor):
        _, C, W, H = img.shape
        output = torch.squeeze(img)
        output = torch.reshape(output, shape=(C, W * H))
        output = torch.transpose(output, 0, 1)
        output = torch.square(output)
        output = torch.matmul(jdx, output)
        output = torch.sqrt(output)
        output = torch.sum(output)
        return output

    @staticmethod
    def superpixel_mean(img: torch.Tensor, jdx: torch.Tensor, jdx_n: torch.Tensor):
        _, C, W, H = img.shape
        output_tensor = torch.squeeze(img)
        output_tensor = torch.reshape(output_tensor, shape=(C, W * H))
        output_tensor = torch.transpose(output_tensor, 0, 1)
        output_tensor = torch.matmul(jdx_n, output_tensor)
        output_tensor = torch.matmul(jdx, output_tensor)
        output_tensor = torch.transpose(output_tensor, 0, 1)
        output_tensor = torch.reshape(output_tensor, shape=(1, C, W, H))
        return output_tensor

    @staticmethod
    def get_param_num(model):
        num = sum(x.numel() for x in model.parameters())
        print("model has {} parameters in total".format(num))
        return num

    @staticmethod
    def to_numpy(val: torch.Tensor):
        return val.cpu().detach().numpy()