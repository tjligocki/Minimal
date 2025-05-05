import torch
import torch.nn as nn
import numpy as np
from skimage import measure
import vtk
from vtk.util import numpy_support

class FourierSurface(nn.Module):
  def __init__(self, order, coeffs=None):
    super().__init__()
    self.order = order
    self.num_coeffs = order**3
    if coeffs is None:
      self.coeffs = nn.Parameter(torch.randn(self.num_coeffs)*0.1)
    else:
      self.coeffs = nn.Parameter(coeffs)

  def forward(self, xyz):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    out = 0.0
    for i in range(self.order):
      for j in range(self.order):
        for k in range(self.order):
          idx = i*self.order**2 + j*self.order + k
          c = self.coeffs[idx]
          out += c*(torch.sin(2*np.pi*i*x)
                   *torch.sin(2*np.pi*j*y)
                   *torch.sin(2*np.pi*k*z))
    return out

def extract_surface(model, device, resolution):
  grid = np.linspace(0,1,resolution)
  xx,yy,zz = np.meshgrid(grid,grid,grid,indexing='ij')
  pts = torch.tensor(
    np.stack([xx,yy,zz],axis=-1),
    dtype=torch.float32, device=device
  )
  pts_flat = pts.reshape(-1,3)
  with torch.no_grad():
    f = model(pts_flat).cpu().numpy().reshape((resolution,)*3)
  verts, faces, *_ = measure.marching_cubes(f, level=0, spacing=(1/resolution,)*3)
  return verts, faces

def export_scalar_fields_to_vti(model, device, resolution, exp_sd, filename):
  import torch
  import numpy as np
  import vtk
  from vtk.util import numpy_support

  # build grid
  grid = np.linspace(0,1,resolution)
  xx,yy,zz = np.meshgrid(grid,grid,grid,indexing='ij')
  pts = torch.tensor(np.stack([xx,yy,zz],axis=-1),
                     dtype=torch.float32, device=device)
  flat = pts.reshape(-1,3).clone().detach().requires_grad_(True)

  f = model(flat)
  grad_f = torch.autograd.grad(f, flat, grad_outputs=torch.ones_like(f),
                               create_graph=True, retain_graph=True)[0]
  norm = grad_f.norm(dim=1,keepdim=True)+1e-8
  ngrad = grad_f / norm

  H = 0
  for i in range(3):
    H += torch.autograd.grad(ngrad[:,i], flat,
                             grad_outputs=torch.ones_like(f),
                             create_graph=True, retain_graph=True)[0][:,i]

  f_np = f.detach().cpu().numpy().reshape((resolution,)*3)
  H_np = H.detach().cpu().numpy().reshape((resolution,)*3)
  W_np = np.exp(-(f_np/exp_sd)**2)

  img = vtk.vtkImageData()
  img.SetDimensions(resolution, resolution, resolution)
  img.SetSpacing(1/resolution,1/resolution,1/resolution)
  img.SetOrigin(0,0,0)

  def _add(name, arr):
    a = numpy_support.numpy_to_vtk(arr.ravel(order='F'), deep=True)
    a.SetName(name)
    img.GetPointData().AddArray(a)

  _add("f", f_np)
  _add("H", H_np)
  _add("W", W_np)

  w = vtk.vtkXMLImageDataWriter()
  w.SetFileName(filename)
  w.SetInputData(img)
  w.Write()

def export_sample_points_with_scalar(xyz, scalar, filename):
  import vtk
  from vtk.util import numpy_support

  pts = xyz.detach().cpu().numpy()
  vals = scalar.detach().cpu().numpy()

  vtk_pts = vtk.vtkPoints()
  vtk_pts.SetData(numpy_support.numpy_to_vtk(pts, deep=True))

  pd = vtk.vtkPolyData()
  pd.SetPoints(vtk_pts)

  arr = numpy_support.numpy_to_vtk(vals, deep=True)
  arr.SetName("S")
  pd.GetPointData().AddArray(arr)

  w = vtk.vtkXMLPolyDataWriter()
  w.SetFileName(filename)
  w.SetInputData(pd)
  w.Write()

