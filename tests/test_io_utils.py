import os
import numpy as np
import torch
from io_utils import save_parameters, load_parameters, save_coefficients_txt, load_coefficients_txt

def test_params_roundtrip(tmp_path):
  p = {"a":1, "b":2.5, "c":[1,2,3]}
  fp = tmp_path/"p.txt"
  save_parameters(p, str(fp))
  q = load_parameters(str(fp))
  assert p==q

def test_coeffs_roundtrip(tmp_path):
  from torch.nn import Parameter
  class M: coeffs=Parameter(torch.arange(8, dtype=torch.float32))
  fp = tmp_path/"c.txt"
  save_coefficients_txt(M, str(fp))
  c2 = load_coefficients_txt(str(fp))
  assert torch.allclose(M.coeffs, c2)

