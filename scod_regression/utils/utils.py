from typing import Optional, Callable

import torch
from torch import Tensor
import numpy as np


def idct(X: Tensor, norm: Optional[str] = None) -> Tensor:
    """
    based on https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
    updated to work with more recent versions of pytorch which moved fft functionality to
    the torch.fft module
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def batch_idct(X: Tensor, norm: Optional[str] = None) -> Tensor:
    """
    based on https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
    updated to work with more recent versions of pytorch which moved fft functionality to
    the torch.fft module
    """
    x_shape = X.shape
    B = x_shape[0]
    N = x_shape[-1]

    X_v = X.contiguous().view(x_shape) / 2

    if norm == "ortho":
        X_v[:, :, 0] *= np.sqrt(N) * 2
        X_v[:, :, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :, :1] * 0, -X_v.flip([2])[:, :, :-1]], dim=2)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(3), V_i.unsqueeze(3)], dim=3)

    v = torch.fft.irfft(torch.view_as_complex(V), n=N, dim=2)

    x = v.new_zeros(v.shape)
    x[:, :, ::2] += v[:, :, : N - (N // 2)]
    x[:, :, 1::2] += v.flip([2])[:, :, : N // 2]

    return x.view(*x_shape)


@torch.no_grad()
def low_rank_approx(Y: Tensor, W: Tensor, Psi_fn: Callable[..., Tensor]):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, M)
    and Psi_fn(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators

    returns Q (N x k), X (k x M) such that A ~= QX
    """
    Q, _ = torch.linalg.qr(Y, "reduced")  # (N, k)
    U, T = torch.linalg.qr(Psi_fn(Q), "reduced")  # (l, k), (k, k)
    X = torch.linalg.solve_triangular(T, U.t() @ W, upper=True)  # (k, N)

    return Q, X


@torch.no_grad()
def fixed_rank_svd_approx(Y: Tensor, W: Tensor, Psi_fn: Callable[..., Tensor], r: int):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, M)
    and Psi_fn(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators

    and a choice of r < k

    returns U (N x r), S (r,), V (M x r) such that A ~= U diag(S) V.T
    """
    Q, X = low_rank_approx(Y, W, Psi_fn)
    U, S, Vh = torch.lianlg.svd(X)
    U = Q @ U[:, :r]

    return U, S[:r], Vh[:r, :]


@torch.no_grad()
def sym_low_rank_approx(Y: Tensor, W: Tensor, Psi_fn: Callable[..., Tensor]):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, N)
    and Psi_fn(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators

    returns U (N x 2k), S (2k x 2k) such that A ~= U S U^T
    """
    Q, X = low_rank_approx(Y, W, Psi_fn)
    k = Q.shape[-1]
    U, T = torch.linalg.qr(torch.cat([Q, X.t()], dim=1), "reduced")  # (N, 2k), (2k, 2k)
    del Q, X
    T1 = T[:, :k]  # (2k, k)
    T2 = T[:, k : 2 * k]  # (2k, k)
    S = (T1 @ T2.t() + T2 @ T1.t()) / 2  # (2k, 2k)

    return U, S


@torch.no_grad()
def fixed_rank_eig_approx(Y: Tensor, W: Tensor, Psi_fn: Callable[..., Tensor], r: int):
    """
    returns U (N x r), D (r) such that A ~= U diag(D) U^T
    """
    U, S = sym_low_rank_approx(Y, W, Psi_fn)
    D, V = torch.linalg.eigh(S)  # (2k), (2k, 2k)
    D = D[-r:]
    V = V[:, -r:]  # (2k, r)
    U = U @ V  # (N, r)
    return U, D
