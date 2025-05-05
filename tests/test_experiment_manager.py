import os
import torch
import numpy as np
from experiment_manager import ExperimentManager
from surface import FourierSurface

def test_save_load_resume(tmp_path):
  em = ExperimentManager(root=str(tmp_path))
  folder = em.get_next_run_folder()
  params = {"fourier_order":2}
  em.save_parameters(params, folder)
  em.save_manifest(folder, {
    "files":{"coefficients":[]},
    "params":params,
    "timestamp":""
  })
  # create a fake model and save coeffs
  surf = FourierSurface(2)
  surf.coeffs.data[:] = 1.234
  em.save_coefficients(surf, folder, step=0)
  # now resume
  surf2, p2, s0 = em.resume_run(folder, FourierSurface)
  assert p2["fourier_order"]==2
  assert s0==0
  assert torch.allclose(surf2.coeffs, surf.coeffs)

