#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from skimage import measure
import pyvista as pv
import sys
import vtk
from vtk.util import numpy_support

def main(argc,argv):
  # Device setup
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device:",device)

  # Parameters
  n_samples = 10000
  fourier_order = 3
  grid_resolution = 64
  training_steps = 1000000
  output_every = 1
  visualize_thres = 0.95
  exp_sd = 0.01

  # Sample random points in the unit cube for training
  xyz = torch.rand(n_samples, 3, device=device, requires_grad=True)

  # Initialize surface model
  surface = FourierSurface(fourier_order).to(device)
  optimizer = torch.optim.Adam(surface.parameters(), lr=1e-2)

  step = 0

  optimizer.zero_grad()

  f_val = surface(xyz)
  grad_f = torch.autograd.grad(f_val, xyz, grad_outputs=torch.ones_like(f_val),
                               create_graph=True, retain_graph=True)[0]

  norm_grad_f = grad_f.norm(dim=1, keepdim=True) + 1e-8
  grad_normalized = grad_f / norm_grad_f

  div = 0
  for i in range(3):
    div += torch.autograd.grad(grad_normalized[:, i], xyz,
                               grad_outputs=torch.ones_like(f_val),
                               create_graph=True, retain_graph=True)[0][:, i]

  mean_curvature = div
  sample_energy = (mean_curvature ** 2) * torch.exp(-(f_val/exp_sd) ** 2)
  energy = sample_energy.mean()
  energy.backward()

  print(f"Step {step}: Energy = {energy.item():.6f}",end="\r")
  sys.stdout.flush()

  save_coefficients_txt(surface, filename=f"coeffs{step}.txt")

  verts, faces = extract_surface(surface, device, resolution=grid_resolution)

  save_surface_as_stl(verts, faces, filename=f"surface_step{step}.stl")
  export_scalar_fields_to_vti(surface, device, grid_resolution, filename=f"fields{step}.vti")
  export_sample_points_with_scalar(xyz, sample_energy, filename=f"sample_points{step}.vtp")

  save_surface_as_stl(verts, faces, filename=f"surface_step{step}.stl")

  old_energy_value = energy.item()

  # Optimization loop
  for step in range(training_steps):
    optimizer.zero_grad()

    f_val = surface(xyz)
    grad_f = torch.autograd.grad(f_val, xyz, grad_outputs=torch.ones_like(f_val),
                                 create_graph=True, retain_graph=True)[0]

    norm_grad_f = grad_f.norm(dim=1, keepdim=True) + 1e-8
    grad_normalized = grad_f / norm_grad_f

    div = 0
    for i in range(3):
      div += torch.autograd.grad(grad_normalized[:, i], xyz,
                                 grad_outputs=torch.ones_like(f_val),
                                 create_graph=True, retain_graph=True)[0][:, i]

    mean_curvature = div
    sample_energy = (mean_curvature ** 2) * torch.exp(-(f_val/exp_sd) ** 2)
    energy = sample_energy.mean()
    energy.backward()
    optimizer.step()

    energy_value = energy.item()

    if step % output_every == 0:
      #print(f"Step {step}: Energy = {energy_value:.6f}")
      print(f"Step {step}: Energy = {energy_value:.6f}",end="\r")
      sys.stdout.flush()

    # if step % visualize_every == 0 and step > 0:

    energy_ratio = energy_value / old_energy_value

    if energy_ratio <= visualize_thres or energy_ratio >= 1.0/visualize_thres:
      print(f"Visualizing and saving at step {step}, Energy = {old_energy_value:.6f} -> {energy_value:.6f} : {energy_ratio:.6f} {visualize_thres*energy_value:.6f}...")
      sys.stdout.flush()

      save_coefficients_txt(surface, filename=f"coeffs{step}.txt")

      verts, faces = extract_surface(surface, device, resolution=grid_resolution)

      save_surface_as_stl(verts, faces, filename=f"surface_step{step}.stl")
      export_scalar_fields_to_vti(surface, device, grid_resolution, filename=f"fields{step}.vti")

      export_sample_points_with_scalar(xyz, sample_energy, filename=f"sample_points{step}.vtp")

      old_energy_value = energy_value

  save_surface_as_stl(verts, faces, filename=f"surface_step{step}.stl")
  export_scalar_fields_to_vti(surface, device, grid_resolution, filename=f"fields{step}.vti")
  save_surface_as_stl(verts, faces, filename=f"surface_step{step}.stl")

def save_coefficients_txt(surface_model, filename="coeffs.txt"):
  coeffs = surface_model.coeffs.detach().cpu().numpy()
  np.savetxt(filename, coeffs.reshape(-1), header=f"{coeffs.shape}", comments="")


def export_scalar_fields_to_vti(f_model, device, resolution=64, filename="fields.vti"):
  # Create a regular grid
  x = np.linspace(0, 1, resolution)
  y = np.linspace(0, 1, resolution)
  z = np.linspace(0, 1, resolution)

  xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

  grid_points = torch.tensor(np.stack([xx, yy, zz], axis=-1), dtype=torch.float32).to(device)
  # grid_flat = grid_points.view(-1, 3).clone().detach().requires_grad_(True)
  grid_flat = grid_points.reshape(-1, 3).clone().detach().requires_grad_(True) 

  f_vals = f_model(grid_flat)

  # Compute grad(f)
  grad_f = torch.autograd.grad(f_vals, grid_flat, grad_outputs=torch.ones_like(f_vals),
                               create_graph=True, retain_graph=True)[0]
  norm_grad_f = grad_f.norm(dim=1, keepdim=True) + 1e-8
  grad_normalized = grad_f / norm_grad_f

  # Compute divergence of normalized gradient = mean curvature
  mean_curv = 0
  for i in range(3):
    mean_curv += torch.autograd.grad(grad_normalized[:, i], grid_flat,
                                     grad_outputs=torch.ones_like(f_vals),
                                     create_graph=True, retain_graph=True)[0][:, i]

  f_vals = f_vals.detach().cpu().numpy().reshape((resolution, resolution, resolution))
  mean_curv = mean_curv.detach().cpu().numpy().reshape((resolution, resolution, resolution))

  # Create VTK image data for VisIt
  image = vtk.vtkImageData()
  image.SetDimensions(resolution, resolution, resolution)
  image.SetSpacing(1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
  image.SetOrigin(0, 0, 0)

  f_array = numpy_support.numpy_to_vtk(num_array=f_vals.ravel(order='F'), deep=True)
  f_array.SetName("f")
  image.GetPointData().AddArray(f_array)

  h_array = numpy_support.numpy_to_vtk(num_array=mean_curv.ravel(order='F'), deep=True)
  h_array.SetName("H")
  image.GetPointData().AddArray(h_array)

  writer = vtk.vtkXMLImageDataWriter()
  writer.SetFileName(filename)
  writer.SetInputData(image)
  writer.Write()

# Fourier-based implicit surface model
#class FourierSurface(nn.Module):
#    def __init__(self, order):
#        super().__init__()
#        self.order = order
#        self.coeffs = nn.Parameter(torch.randn((order, order, order), device=device) * 0.1)
#
#    def forward(self, xyz):
#        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
#        f = torch.zeros_like(x)
#        for i in range(self.order):
#            for j in range(self.order):
#                for k in range(self.order):
#                    a = self.coeffs[i, j, k]
#                    f += a * torch.sin(2 * np.pi * (i+1) * x) * \
#                             torch.sin(2 * np.pi * (j+1) * y) * \
#                             torch.sin(2 * np.pi * (k+1) * z)
#        return f


class FourierSurface(nn.Module):
  def __init__(self, order, coeffs=None):
    super().__init__()
    self.order = order
    self.num_coeffs = order ** 3
    if coeffs is None:
      self.coeffs = nn.Parameter(torch.randn(self.num_coeffs) * 0.1)
    else:
      self.coeffs = nn.Parameter(coeffs)

  def forward(self, xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    result = 0.0
    for i in range(self.order):
      for j in range(self.order):
        for k in range(self.order):
          idx = i * self.order**2 + j * self.order + k
          coeff = self.coeffs[idx]
          result += coeff * (torch.sin(2 * np.pi * i * x)
                           * torch.sin(2 * np.pi * j * y)
                           * torch.sin(2 * np.pi * k * z))
    return result


# Marching Cubes surface extraction
def extract_surface(f_model, device, resolution=64):
  x = np.linspace(0, 1, resolution)
  y = np.linspace(0, 1, resolution)
  z = np.linspace(0, 1, resolution)

  xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

  grid = torch.tensor(np.stack([xx, yy, zz], axis=-1), dtype=torch.float32).to(device)
  grid_flat = grid.reshape(-1, 3)

  with torch.no_grad():
    f_vals = f_model(grid_flat).cpu().numpy().reshape((resolution,)*3)

  verts, faces, normals, _ = measure.marching_cubes(f_vals, level=0.0, spacing=(1/resolution,)*3)

  return verts, faces


# Visualization with PyVista
def visualize_surface(verts, faces):
  mesh = pv.PolyData(verts, np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32))
  plotter = pv.Plotter()
  plotter.add_mesh(mesh, color="lightblue", show_edges=False)
  plotter.show_grid()
  plotter.show()


def save_surface_as_stl(verts, faces, filename="surface.stl"):
  # Convert vertices and faces into a PyVista mesh and save as STL
  faces_stl = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
  mesh = pv.PolyData(verts, faces_stl)
  mesh.save(filename)


def export_sample_points_with_scalar(xyz, scalar, filename="sample_points.vtp"):
    import vtk
    from vtk.util import numpy_support

    xyz_np = xyz.detach().cpu().numpy()
    val_np = scalar.detach().cpu().numpy()

    # Create VTK Points object
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(xyz_np, deep=True))

    # Create VTK PolyData to hold points
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Add mean curvature as a point scalar
    s_array = numpy_support.numpy_to_vtk(val_np, deep=True)
    s_array.SetName("S")
    polydata.GetPointData().AddArray(s_array)

    # Write to VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


if __name__ == "__main__":
  argv = sys.argv
  argc = len(argv)

  main(argc,argv)
