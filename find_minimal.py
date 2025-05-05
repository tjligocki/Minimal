#!/usr/bin/env python3

import sys
import os
import torch
import random
from datetime import datetime

from experiment_manager import ExperimentManager
from io_utils        import save_parameters, load_parameters
from surface         import (FourierSurface,
                             extract_surface,
                             export_scalar_fields_to_vti,
                             export_sample_points_with_scalar)

def main(argc, argv):
  if argc>1 and argv[1]=="test":
    import pytest
    pytest.main([os.path.dirname(__file__)+"/tests"])
    return

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  mgr = ExperimentManager()

  if argc>1:
    folder=argv[1]
    surface, params, start = mgr.resume_run(folder, FourierSurface)
  else:
    folder = mgr.get_next_run_folder()
    params = {
      "n_samples":10000,
      "fourier_order":3,
      "grid_resolution":64,
      "training_steps":100000,
      "output_every":100,
      "visualize_thres":0.95,
      "exp_sd":0.01,
      "lr":1e-2
    }
    save_parameters(params, os.path.join(folder,"params.txt"))
    surface = FourierSurface(params["fourier_order"]).to(device)
    start = 0

  xyz = torch.rand(params["n_samples"],3,device=device,requires_grad=True)
  opt = torch.optim.Adam(surface.parameters(),lr=params["lr"])
  old_e = None

  for step in range(start, params["training_steps"]):
    opt.zero_grad()
    f    = surface(xyz)
    gf   = torch.autograd.grad(f, xyz, grad_outputs=torch.ones_like(f),
                               create_graph=True, retain_graph=True)[0]
    norm = gf.norm(dim=1,keepdim=True)+1e-8
    ng   = gf/norm

    div = 0
    for i in range(3):
      div += torch.autograd.grad(ng[:,i], xyz,
                                 grad_outputs=torch.ones_like(f),
                                 create_graph=True, retain_graph=True)[0][:,i]

    H   = div
    e_s = (H**2)*torch.exp(-(f/params["exp_sd"])**2)
    e   = e_s.mean()
    e.backward()
    opt.step()
    ev = e.item()

    if step%params["output_every"]==0:
      print(f"Step {step}: {ev:.6f}", end="\r")

    if old_e is None:
      old_e = ev

    ratio = ev/old_e
    if ratio<=params["visualize_thres"] or ratio>=1/params["visualize_thres"]:
      print(f"\nSaving at step {step}  ({old_e:.4f}->{ev:.4f})")
      mgr.save_coefficients(surface, folder, step)
      mgr.save_energy(folder, ev, step)
      export_scalar_fields_to_vti(surface, device,
                                  params["grid_resolution"],
                                  params["exp_sd"],
                                  os.path.join(folder,f"fields_step{step:05d}.vti"))
      export_sample_points_with_scalar(xyz, e_s,
                                os.path.join(folder,f"points_step{step:05d}.vtp"))
      old_e = ev

if __name__=="__main__":
  argc = len(sys.argv)
  argv = sys.argv
  main(argc, argv)

