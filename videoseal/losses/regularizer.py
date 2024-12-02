
import torch
import torch.nn as nn
import torch.distributions as dist


class MMDLoss(nn.Module):
    def __init__(self, mean=0.0, std=1.0, kernel='rbf'):
        super(MMDLoss, self).__init__()
        self.mean = mean
        self.std = std
        self.kernel = kernel
        self.gaussian = dist.Normal(mean, std)

    def forward(
        self, 
        imgs: torch.Tensor,
        imgs_w: torch.Tensor,
    ) -> torch.Tensor:
        deltas = imgs_w - imgs  # b c h w
        bsz, ch, height, width = deltas.size()  # b c h w
        target = self.gaussian.sample((bsz, ch, height, width)).to(deltas.device)
        return self.mmd_loss(deltas.view(bsz, -1), target.view(bsz, -1))

    def mmd_loss(self, deltas, target):
        XX = torch.matmul(deltas, deltas.t())
        YY = torch.matmul(target, target.t())
        XY = torch.matmul(deltas, target.t())

        X2 = torch.sum(deltas * deltas, dim=1, keepdim=True)
        Y2 = torch.sum(target * target, dim=1, keepdim=True)

        distances = X2 + Y2.t() - 2 * XY

        if self.kernel == 'rbf':
            bandwidth = torch.median(distances)
            bandwidth = bandwidth.item()
            scale = 2 * bandwidth * bandwidth
            kernel_val = torch.exp(-distances / scale)
        else:
            raise ValueError("Unsupported kernel")

        return torch.mean(XX + YY - 2 * XY)



class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
    
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


class WassersteinLoss(nn.Module):
    def __init__(self, reg=0.01, num_iter=100, mean=0.0, std=1.0):
        super(WassersteinLoss, self).__init__()
        self.reg = reg
        self.num_iter = num_iter
        self.gaussian = dist.Normal(mean, std)

    def sinkhorn(self, source, target, cost_matrix):
        n = source.size(0)
        m = target.size(0)

        # Uniform distribution on samples
        a = torch.full((n,), fill_value=1/n, dtype=torch.float32, device=source.device)
        b = torch.full((m,), fill_value=1/m, dtype=torch.float32, device=target.device)

        # Initialize dual variables
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        # Sinkhorn iterations
        for _ in range(self.num_iter):
            u = self.reg * (torch.log(a) - torch.logsumexp(self.M(cost_matrix, u, v), dim=1))
            v = self.reg * (torch.log(b) - torch.logsumexp(self.M(cost_matrix, u, v).transpose(0, 1), dim=1))

        # Compute the Wasserstein distance using the dual variables
        wasserstein_distance = torch.sum(u * a + v * b)
        return wasserstein_distance

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        return C - u.unsqueeze(1) - v.unsqueeze(0)

    def wasserstein_loss(self, source, target):
        # Compute the cost matrix
        cost_matrix = torch.cdist(source, target, p=2)  # Euclidean distance

        # Sinkhorn algorithm for approximate Wasserstein distance
        return self.sinkhorn(source, target, cost_matrix)

    def forward(
        self, 
        imgs: torch.Tensor,
        imgs_w: torch.Tensor,
    ) -> torch.Tensor:
        deltas = imgs_w - imgs
        bsz, ch, height, width = deltas.size()  # b c h w
        target = self.gaussian.sample((bsz, ch, height, width)).to(deltas.device)
        return self.wasserstein_loss(deltas.view(deltas.size(0), -1), target.view(target.size(0), -1))

