import os
import json
import random
import torch
import numpy as np
from datetime import datetime

class ExperimentManager:
  def __init__(self, root="experiments"):
    self.root = root
    os.makedirs(self.root, exist_ok=True)

  def get_next_run_folder(self):
    existing = [d for d in os.listdir(self.root) if d.startswith('run_')]
    nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    nxt = max(nums) + 1 if nums else 1
    folder = os.path.join(self.root, f"run_{nxt:04d}")
    os.makedirs(folder)
    return folder

  def save_parameters(self, params, folder):
    path = os.path.join(folder, "params.txt")
    with open(path, 'w') as f:
      for k, v in params.items():
        f.write(f"{k} = {v}\n")

  def load_parameters(self, folder):
    import ast
    params = {}
    path = os.path.join(folder, "params.txt")
    with open(path, 'r') as f:
      for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
          continue
        k, v = line.split('=', 1)
        params[k.strip()] = ast.literal_eval(v.strip())
    return params

  def save_manifest(self, folder, manifest):
    path = os.path.join(folder, "manifest.json")
    with open(path, 'w') as f:
      json.dump(manifest, f, indent=2)

  def load_manifest(self, folder):
    path = os.path.join(folder, "manifest.json")
    if not os.path.exists(path):
      return {}
    with open(path, 'r') as f:
      return json.load(f)

  def update_manifest_files(self, folder, filetype, filename):
    m = self.load_manifest(folder)
    if "files" not in m:
      m["files"] = {"fields": [], "points": [], "coefficients": []}
    m["files"].setdefault(filetype, []).append(filename)
    self.save_manifest(folder, m)

  def save_coefficients(self, model, folder, step):
    coeffs = model.coeffs.detach().cpu().numpy()
    name = f"coeffs_step{step:05d}.txt"
    path = os.path.join(folder, name)
    with open(path, 'w') as f:
      f.write(f"{coeffs.shape}\n")
      np.savetxt(f, coeffs.reshape(-1))
    self.update_manifest_files(folder, "coefficients", name)

  def save_energy(self, folder, energy, step):
    import csv
    path = os.path.join(folder, "energy_log.csv")
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
      w = csv.writer(f)
      if new:
        w.writerow(["step", "energy"])
      w.writerow([step, energy])

  def resume_run(self, folder, surface_cls):
    import ast
    params = self.load_parameters(folder)
    m = self.load_manifest(folder)
    coeffs_list = m.get("files", {}).get("coefficients", [])
    if not coeffs_list:
      raise RuntimeError("no coefficients to resume from")
    latest = sorted(coeffs_list)[-1]
    step = int(latest.split('step')[1].split('.')[0])
    path = os.path.join(folder, latest)
    with open(path, 'r') as f:
      shape = ast.literal_eval(f.readline())
      data = np.array([float(x) for x in f.readlines()])
    coeffs = torch.tensor(data, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surface = surface_cls(order=params["fourier_order"], coeffs=coeffs).to(device)
    print(f"✅ resumed {folder} at step {step}")
    return surface, params, step

