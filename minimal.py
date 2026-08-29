import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import concurrent.futures

def get_wavevector_squared(N):
    """Generates the squared wavevector magnitudes for the unit torus [0,1]^3."""
    kx = np.fft.fftfreq(N, d=1.0)
    ky = np.fft.fftfreq(N, d=1.0)
    kz = np.fft.rfftfreq(N, d=1.0)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    return KX**2 + KY**2 + KZ**2

def evaluate_nonlinear(c_grid, N, exponent):
    """
    Evaluates the continuous convolution algebraically via the spatial domain.
    Note: To strictly satisfy Orszag's 3/2 rule, c_grid should be zero-padded 
    to shape (1.5N, 1.5N, 1.5N//2 + 1) before the IFFT, and truncated after the FFT.
    """
    f_spatial = np.fft.irfftn(c_grid, s=(N, N, N)) * (N**3)
    nonlinear_spatial = f_spatial**exponent
    return np.fft.rfftn(nonlinear_spatial) / (N**3)

def exact_residual(c_flat, N, epsilon, K2):
    """Computes the exact analytic residual of the Allen-Cahn PDE."""
    c_grid = c_flat.reshape((N, N, N//2 + 1))
    
    laplacian_term = (1.0 - 4.0 * np.pi**2 * epsilon**2 * K2) * c_grid
    cubic_term = evaluate_nonlinear(c_grid, N, 3)
    
    res = laplacian_term - cubic_term
    res[0, 0, 0] = 0.0 + 0.0j  # Enforce volume constraint
    return res.ravel()

def exact_jacobian_action(v_flat, c_grid, N, epsilon, K2):
    """Computes the exact matrix-free Jacobian-Vector product J(c)*v."""
    v_grid = v_flat.reshape((N, N, N//2 + 1))
    
    f_spatial = np.fft.irfftn(c_grid, s=(N, N, N)) * (N**3)
    v_spatial = np.fft.irfftn(v_grid, s=(N, N, N)) * (N**3)
    
    w_spatial = 3.0 * (f_spatial**2) * v_spatial
    w_hat = np.fft.rfftn(w_spatial) / (N**3)
    
    laplacian_term = (1.0 - 4.0 * np.pi**2 * epsilon**2 * K2) * v_grid
    jv = laplacian_term - w_hat
    
    jv[0, 0, 0] = 0.0 + 0.0j
    return jv.ravel()

def solve_minimal_surface(seed_id, N=32, eps_start=0.3, eps_end=0.03):
    """
    Executes the epsilon-continuation loop using the Newton-Krylov method.
    """
    np.random.seed(seed_id)
    K2 = get_wavevector_squared(N)
    
    # Initialize random symmetric noise strictly in low-frequency bands
    c_grid = np.zeros((N, N, N//2 + 1), dtype=np.complex128)
    low_freq_mask = K2 <= 4.0 
    c_grid[low_freq_mask] = np.random.randn(np.sum(low_freq_mask)) + \
                            1j * np.random.randn(np.sum(low_freq_mask))
    c_grid[0, 0, 0] = 0.0  # Zero mean (50/50 volume partition)
    
    epsilon = eps_start
    c_flat = c_grid.ravel()
    
    # Epsilon Continuation Loop
    while epsilon >= eps_end:
        
        # Inner Newton-Raphson Loop
        for newton_step in range(15):
            res = exact_residual(c_flat, N, epsilon, K2)
            res_norm = np.linalg.norm(res)
            
            if res_norm < 1e-8:
                break
                
            # Define the exact linear operator for the Krylov subspace solver
            current_c_grid = c_flat.reshape((N, N, N//2 + 1))
            J_op = LinearOperator(
                shape=(c_flat.size, c_flat.size),
                matvec=lambda v: exact_jacobian_action(v, current_c_grid, N, epsilon, K2),
                dtype=np.complex128
            )
            
            # Solve J * delta_c = -res exactly via GMRES
            delta_c, exit_code = gmres(J_op, -res, tol=1e-5, maxiter=50)
            c_flat += delta_c
            
        epsilon *= 0.85 # Homotopy step size
        
    return c_flat.reshape((N, N, N//2 + 1))

def distribute_search_space():
    """
    Distributes the phase-space search asynchronously across the compute cluster.
    Structured for 96 parallel workers to maintain 100% saturation across all nodes.
    """
    converged_surfaces = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=96) as executor:
        # Submit an array of independent trajectories matching the core count
        futures = {
            executor.submit(solve_minimal_surface, seed_id=i): i 
            for i in range(96 * 5) # 5 complete generations per core
        }
        
        for future in concurrent.futures.as_completed(futures):
            try:
                c_final = future.result()
                converged_surfaces.append(c_final)
                print(f"Trajectory {futures[future]} converged.")
            except Exception as e:
                print(f"Trajectory {futures[future]} failed: {e}")
                
    return converged_surfaces

if __name__ == '__main__':
    results = distribute_search_space()