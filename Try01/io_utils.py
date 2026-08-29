import numpy as np
import torch
import ast

def save_parameters(params, filename="params.txt"):
  with open(filename, 'w') as f:
    for k, v in params.items():
      f.write(f"{k} = {v}\n")

def load_parameters(filename="params.txt"):
  params = {}
  with open(filename, 'r') as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      k, v = line.split('=', 1)
      params[k.strip()] = ast.literal_eval(v.strip())
  return params

def save_coefficients_txt(model, filename="coeffs.txt"):
  coeffs = model.coeffs.detach().cpu().numpy()
  with open(filename, 'w') as f:
    f.write(f"{coeffs.shape}\n")
    np.savetxt(f, coeffs.reshape(-1))

def load_coefficients_txt(filename):
  with open(filename, 'r') as f:
    header = f.readline().strip()
    shape = ast.literal_eval(header)
    data = np.loadtxt(f)
  return torch.tensor(data, dtype=torch.float32)

