import torch
import numpy as np
from surface import FourierSurface, extract_surface, export_scalar_fields_to_vti

def test_sphere_mean_curvature(tmp_path):
  R = 0.5
  # define sphere f(x)=||x||-R
  class S(torch.nn.Module):
    def forward(self, xyz): return xyz.norm(dim=1)-R

  # sample a few points
  pts = torch.rand(20000,3,requires_grad=True)
  f = S()(pts)
  gf = torch.autograd.grad(f, pts, grad_outputs=torch.ones_like(f),
                           create_graph=True)[0]
  norm = gf.norm(dim=1,keepdim=True)+1e-8
  n = gf/norm
  H = 0
  for i in range(3):
    H += torch.autograd.grad(n[:,i], pts,
                             grad_outputs=torch.ones_like(f),
                             create_graph=True)[0][:,i]
  mean_H = H.mean().item()
  assert abs(mean_H-2/R) < 1e-2

