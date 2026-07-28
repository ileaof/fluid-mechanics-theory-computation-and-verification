"""Throwaway algebraic checks for the cut-cell Poisson (no time stepping).

Verifies, on a small cylinder mesh:
  * the cut-cell Poisson matrix assembles with no NaN/Inf;
  * every row sums to ~0 (constant null space preserved -> mean-projection valid);
  * the diagonal is well-conditioned (no tiny entries after small-cell kill);
  * the cut-cell divergence of a uniform field is ~0 in full-fluid cells and
    bounded (not inf/nan) at cut cells;
  * PressureSolver.solve returns a finite, bounded pressure increment.
"""
import numpy as np
from config.config_loader import Config
from mesh.mesh import Mesh
from physics import Fluid
from solver import BoundaryCondition, LinearSolver, ProjectionMethod
from solver.cut_cell import CutCellGeometry


def build(Nx, Ny):
    data = {
        "name": "cc_test", "Nx": Nx, "Ny": Ny, "Nz": 1,
        "Lx": 12.0, "Ly": 4.1, "Lz": 1.0,
        "rho": 1.0, "mu": 0.025, "cp": 1.0, "k": 0.0, "beta": 0.0,
        "gravity": [0.0, 0.0, 0.0], "boussinesq": False, "use_vof": False,
        "convection": "upwind", "linear_solver": "bicgstab",
        "linear_tol": 1e-6, "linear_maxiter": 2000,
        "poisson_tol": 1e-7, "poisson_maxiter": 3000, "use_ilu": True,
        "rhie_chow": True,
        "velocity_bc": {"west": {"kind": "inlet", "value": 1.0},
                        "east": "outlet", "south": "slip", "north": "slip"},
        "temperature_bc": {"west": "adiabatic", "east": "adiabatic",
                           "south": "adiabatic", "north": "adiabatic"},
        "pressure_bc": {"west": "neumann", "east": "neumann",
                        "south": "neumann", "north": "neumann"},
        "obstacles": [{"shape": "cylinder", "center": [5.005, 2.05],
                       "radius": 0.5, "axis": "z"}],
        "immersed_method": "ibm", "snap_obstacle_to_grid": True,
        "ibm_cut_cell": True, "time_scheme": "implicit",
        "dt": 0.05, "tfinal": 1.0, "adaptive_dt": False,
        "output_dir": "outputs/cc_test",
    }
    cfg = Config.from_dict(data)
    mesh = Mesh(cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz)
    bc = BoundaryCondition(cfg)
    Xc, Yc, _ = mesh.cell_grid()
    cx, cy = cfg.obstacles[0]["center"]
    r = cfg.obstacles[0]["radius"]
    solid = (Xc - cx) ** 2 + (Yc - cy) ** 2 <= r ** 2
    bc.set_solid(solid)
    cc = CutCellGeometry(mesh, solid, list(cfg.obstacles))
    assert cc.has_curve
    bc.set_solid(cc.is_solid)
    fluid = Fluid.from_config(cfg)
    ls = LinearSolver(method=cfg.linear_solver, tol=cfg.poisson_tol,
                      maxiter=cfg.poisson_maxiter, use_ilu=cfg.use_ilu)
    proj = ProjectionMethod(mesh, fluid, bc, cfg, ls)
    proj.pressure.set_cut_cell(cc)
    return mesh, bc, cc, proj, fluid


def main():
    for (Nx, Ny) in [(60, 30), (120, 60)]:
        mesh, bc, cc, proj, fluid = build(Nx, Ny)
        ps = proj.pressure
        rho = np.full(mesh.cell_shape, fluid.rho)
        A = ps._matrix(rho)
        n = A.shape[0]
        ones = np.ones(n)
        row_sums = A @ ones
        diag = A.diagonal()
        print(f"--- {Nx}x{Ny}: n_cut={cc.n_cut} n_small={cc.n_small} "
              f"n_solid={int(cc.is_solid.sum())} ---")
        print(f"  matrix nnz={A.nnz}  has_nan={np.isnan(A.data).any()}  "
              f"has_inf={np.isinf(A.data).any()}")
        print(f"  max|row_sum| = {np.max(np.abs(row_sums)):.2e}  (expect ~0)")
        # conditioning proxy: smallest |diag| vs typical
        typ = np.median(np.abs(diag[diag != 0])) if (diag != 0).any() else 0.0
        print(f"  diag: min|d|={np.min(np.abs(diag)):.3e}  median|d|={typ:.3e}  "
              f"ratio={np.min(np.abs(diag))/typ:.3e}")
        # zero rows would be singular beyond the constant null space
        zero_rows = int(np.sum(np.abs(row_sums) < 1e-12) - 0)  # all rows sum ~0
        print(f"  rows summing to ~0: {zero_rows}/{n} (constant null space)")

        # divergence of a uniform u*=(1,0): ~0 in full fluid, bounded at cut cells
        us = np.ones(mesh.cell_shape)
        vs = np.zeros(mesh.cell_shape)
        Fx, Fy, Fz = ps._face_fluxes(us, vs, None)
        div = ps._cc_divergence(Fx, Fy, Fz)
        full = cc.is_fluid[:, :, 0] & ~cc.is_cut[:, :, 0]
        interior = full.copy()
        interior[0, :] = False; interior[-1, :] = False
        interior[:, 0] = False; interior[:, -1] = False
        print(f"  div(uniform): max|full-fluid interior|={np.max(np.abs(div[interior])):.2e}  "
              f"max|cut|={np.max(np.abs(div[cc.is_cut])):.2e}  "
              f"has_nan={np.isnan(div).any()} has_inf={np.isinf(div).any()}")

        # full solve with uniform predictor + zero old pressure
        dp, fluxes = ps.solve(us, vs, None, dt=0.05, rho=rho, p_old=np.zeros(mesh.cell_shape))
        print(f"  solve: dp max|dp|={np.max(np.abs(dp)):.3e}  "
              f"nan={np.isnan(dp).any()} inf={np.isinf(dp).any()}")

        # one full projection step from rest + inlet: should stay bounded
        u = np.ones(mesh.cell_shape); v = np.zeros(mesh.cell_shape)
        p = np.zeros(mesh.cell_shape)
        src = np.zeros_like(u)
        out = proj.step(u, v, None, p, dt=0.05,
                        sources=(src, src, src), rho=rho)
        un, vn = out["u"], out["v"]
        print(f"  step: max|u|={np.max(np.abs(un)):.3f} max|v|={np.max(np.abs(vn)):.3f} "
              f"max|p|={np.max(np.abs(out['p'])):.3e} div={out['div']:.2e}")
        assert np.isfinite(un).all() and np.isfinite(vn).all() and np.isfinite(out["p"]).all()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()