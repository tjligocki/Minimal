import torch
import torch.nn as nn
from surface import FourierSurface  # for type reference only
from torch import autograd
import numpy as np

def test_sphere_mean_curvature():
  R = 0.5

  # Implicit sphere f(x) = ‖x‖ − R
  class SphereSurface(nn.Module):
    def forward(self, xyz):
      return xyz.norm(dim=1) - R

  # Sample points *on* the sphere of radius R
  N = 20000
  pts = torch.randn(N, 3)
  pts = pts / pts.norm(dim=1, keepdim=True) * R
  pts.requires_grad_(True)

  # Compute f and its gradient
  f = SphereSurface()(pts)
  grad_f = autograd.grad(f, pts, grad_outputs=torch.ones_like(f),
                         create_graph=True)[0]

  # Normalize gradient to get surface normal direction
  norm_grad = grad_f.norm(dim=1, keepdim=True) + 1e-8
  n = grad_f / norm_grad

  # Compute divergence of the normal = mean curvature
  H = torch.zeros_like(f)
  for i in range(3):
    H = H + autograd.grad(n[:, i], pts,
                          grad_outputs=torch.ones_like(f),
                          create_graph=True)[0][:, i]

  mean_H = H.mean().item()
  expected = 2.0 / R
  assert abs(mean_H - expected) < 1e-2

