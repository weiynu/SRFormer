import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random

# -----------------------
# WBCP inner optimization: for a given batch of token embeddings, fix model params,
# optimize per-sample 'ratio' to maximize entropy while keeping outputs close.
# Returns sigma and attention supervision alpha_tilde per sample.
# -----------------------
def wbcp_inner_optimize(embeddings,
                        net1,
                        net2,
                        inner_steps=40,
                        lr_inner=0.01,
                        rate1=0.1,          # entropy weight
                        rate2=1.0,          # auxiliary loss weight
                        device='mps' if torch.backends.mps.is_available() else 'cpu'):
    """
    embeddings: typically model.embed(input_ids) or precomputed embeddings (B, T, N, D)
    We'll create a ratio parameter per sample: shape (B,T, N, 1), optimize it (requires_grad).
    """

    B, T, N, D = embeddings.shape #[B,T,N,D]
    # scale = embeddings.std().item() * 10.0 + 1e-8  # the variance of embeddings
    scale = embeddings.std().item() + 1e-8

    # create ratio var (not registered as module param) with requires_grad
    ratio = torch.randn(B, T, N, 1, device=device, requires_grad=True) #[B,T,N,1]
    ratio = nn.Parameter(ratio * 0.01)

    opt = Adam([ratio], lr=lr_inner)

    eps = 1e-8

    net1.to(device)
    net2.to(device)
    net1.train()
    net2.train()

    for step in range(inner_steps):
        opt.zero_grad()
        ratios = torch.sigmoid(ratio)  # (B,T,N,1)
        # sample noise ~ N(0,1), shape (B,S,D)
        x = embeddings
        noise = torch.randn_like(embeddings, device=device)
        x_noise = x + ratios * noise * scale  # (B,T,N,D)

        # forward through networks
        x = x.transpose(1, 2)
        x=x.reshape(B,N,T*D)

        x_noise = x_noise.transpose(1, 2)
        x_noise=x_noise.reshape(B,N,T*D)

        y = net1(x).view(B,N,T,1)
        y_noise = net1(x_noise).view(B,N,T,1)
        yl = net2(x).view(B,N,T,1)
        y_noisel = net2(x_noise).view(B,N,T,1)

        # use logits directly for MSE
        loss = F.mse_loss(y_noise, y, reduction='none').mean(dim=1)  # (B,)
        lossl = F.mse_loss(y_noisel, yl, reduction='none').mean(dim=1)  # (B,)

        # normalization by mean square of y (like TF code)
        # compute mean square per sample
        denom_y = (y ** 2).mean(dim=1).detach() + eps
        denom_yl = (yl ** 2).mean(dim=1).detach() + eps
        loss = (loss / denom_y).mean()
        lossl = (lossl / denom_yl).mean()

        entropy_term = - torch.mean(torch.log(ratios + eps))  # negative because we want to maximize entropy -> minimize -log(ratio)
        total = loss + rate2 * lossl + (-rate1) * (-entropy_term)  # note: paper uses -mean(log(ratios)) * rate1 as addition -> implemented below more directly


        total.backward()
        opt.step()

    # after optimization, compute sigma and attention supervision α̃
    with torch.no_grad():
        ratios = torch.sigmoid(ratio)  # (B,T,N,1)
        # sigma = (ratios[:, :, 0] * scale).detach()  # (B, S)
        # compute alpha' = 1 - sigma / max_j sigma_j (per sample)
        max_sigma = ratios.max(dim=-1, keepdim=True)[0] + eps  # (B,1)
        alpha_prime = 1.0 - ratios / max_sigma  # (B,S)
        # mask out padding positions by setting very negative before softmax
        alpha_tilde = F.softmax(alpha_prime, dim=-1)  # (B,S)
    return ratios, alpha_tilde