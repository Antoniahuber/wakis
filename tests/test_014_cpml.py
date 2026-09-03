import os
import sys

import numpy as np
import pyvista as pv
from scipy.constants import c, mu_0
from tqdm import tqdm

import pytest

sys.path.append("../wakis")
import wakis

flag_interactive = False  # Set to true to run plot tests


class TestCPML:
    """Test CPML implementation in SolverFIT3D. First test is a reflection test with a Gaussian packet, 
    second test is a TFSF simulation of a cubic cavity. It benchmarks the impedance of the cavity against 
    the current simulation results with CPML and TF/SF as reference."""

    Zabs = np.array([4.03316939e+00, 6.40944766e+00, 2.19024949e+00, 1.21880622e+01,
        1.23294143e+01, 1.27499956e+01, 2.28711435e+01, 1.76265519e+01,
        2.55719674e+01, 3.21556326e+01, 2.44677336e+01, 4.00229861e+01,
        3.94270613e+01, 3.54021504e+01, 5.52806839e+01, 4.47593783e+01,
        5.19186925e+01, 7.02621271e+01, 4.97280949e+01, 7.45381000e+01,
        8.35855430e+01, 5.88868691e+01, 1.03468478e+02, 9.35430277e+01,
        8.00313672e+01, 1.39003244e+02, 9.83238193e+01, 1.21880354e+02,
        1.81898026e+02, 9.80683283e+01, 1.95991840e+02, 2.34375712e+02,
        1.08734007e+02, 3.30687682e+02, 3.04220947e+02, 2.10965682e+02,
        6.36300123e+02, 4.32885394e+02, 7.58922268e+02, 2.33171035e+03,
        3.31244177e+03, 2.97236055e+03, 1.52721811e+03, 1.95981363e+02,
        7.93246456e+02, 5.28129975e+02, 2.83132056e+02, 5.14058281e+02,
        3.23514165e+02, 3.25904302e+02])
    
    def test_reflection_gaussianPacket(self, use_gpu):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = 0.0, 1.0

        Nx, Ny = 8, 8
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

        # Boundary conditions and solver
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "cpml"]

        # Test different eps_r and sigma case
        eps_r = 1.0
        sigma = 0.0

        # Solver
        solver = wakis.SolverFIT3D(
            grid,
            use_stl=False,
            use_gpu=use_gpu,
            bg=[eps_r, 1.0, sigma],
            bc_low=bc_low,
            bc_high=bc_high,
            n_pml=8,
            kappa_max=5,
            alpha_max=0.05,
            sigma_factor=1,
            pml_exp=4,
            dtype=np.float32,
        )

        # Source
        amplitude = 1.
        gaussianPacket = wakis.sources.GaussianPacket(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            sigmaz=15e-3,
            sigmaxy=100.,
            amplitude=amplitude,
        )

        Nt = int(gaussianPacket.tinj+2.0*(zmax-zmin)/c/solver.dt)
        forward = int((gaussianPacket.tinj+0.5*(zmax-zmin))/c/solver.dt)
        backward = int((gaussianPacket.tinj+1.5*(zmax-zmin))/c/solver.dt)

        for n in tqdm(range(Nt)):
            gaussianPacket.update(solver, n * solver.dt)
            solver.one_step()
            if n == forward:
                Exfor = solver.E[Nx//2, Ny//2, :-solver.n_pml, 'x'].copy()
            if n == backward:
                Exback = solver.E[Nx//2, Ny//2, :-solver.n_pml, 'x'].copy()

            if flag_interactive and n % int(Nt / 100) == 0:
                solver.plot1D(
                    "Hy",
                    ylim=(-amplitude, amplitude),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Hy",
                    n=n,
                )
                solver.plot1D(
                    "Ex",
                    ylim=(-amplitude * c * mu_0, amplitude * c * mu_0),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Ex",
                    n=n,
                )

        reflection_factor = (np.abs(Exback).max()/np.abs(Exfor).max())**2
        assert reflection_factor <= 1e-6, (
            f"CPML Ex reflection factor in average > 1e-6 with eps_r={eps_r}, sigma={sigma}, reflection_factor={reflection_factor}"
        )

        t = solver.z[:-solver.n_pml] /c
        Sfor = np.abs(np.fft.fft(Exfor))
        Sback = np.abs(np.fft.fft(Exback))
        S = (Sback / Sfor)**2
        f = np.fft.fftfreq(len(t), d=t[1] - t[0])
        mask = (0 <= f) & (f <= 6.66e9)

        assert S[mask].max() <= 1e-4, (
            f"Maximal CPML Ex reflection factor over all frequencies >1e-4 with eps_r={eps_r}, sigma={sigma}, Smax={S[mask].max()}"
        )

        if flag_interactive:

            solver.plot2D(
                "Ex",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude * c * mu_0,
                vmax=amplitude * c * mu_0,
                off_screen=True,
                title="005_Ex2d",
            )

            solver.plot2D(
                "Hy",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude,
                vmax=amplitude,
                off_screen=True,
                title="005_Hy2d",
            )

    def test_tfsf_simulation(self, use_gpu):
        print("\n---------- Initializing simulation ------------------")
        # Number of mesh cells
        Nx = 50
        Ny = 50
        Nz = 150

        # Embedded boundaries
        stl_file = "tests/stl/001_cubic_cavity.stl"
        surf = pv.read(stl_file)

        stl_solids = {"cavity": stl_file}
        stl_materials = {"cavity": "vacuum"}

        # Domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

        # set grid and geometry
        global grid
        grid = wakis.GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            Nx,
            Ny,
            Nz,
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            verbose=2,
        )

        # Beam parameters
        beta = 1.0  # beam beta
        sigmaz = 18.5e-3 * beta  # [m]
        q = 1e-9  # [C]
        xs = 0.0  # x source position [m]
        ys = 0.0  # y source position [m]
        xt = 0.0  # x test position [m]
        yt = 0.0  # y test position [m]

        global wake
        skip_cells = 8  # no. cells to skip in WP integration
        wakelength = 1.0  # [m]
        wake = wakis.WakeSolver(
            wakelength=wakelength,
            q=q,
            sigmaz=sigmaz,
            beta=beta,
            xsource=xs,
            ysource=ys,
            xtest=xt,
            ytest=yt,
            save=False,
            Ez_file="tests/014_Ez.h5",
            skip_cells=skip_cells,
            verbose=2,
        )

        # boundary conditions
        bc_low = ["pec", "pec", "cpml"]
        bc_high = ["pec", "pec", "cpml"]

        # set Solver object
        solver = wakis.SolverFIT3D(
            grid,
            wake,
            bc_low=bc_low,
            bc_high=bc_high,
            use_stl=True,
            bg="pec",
            dtype=np.float32,
            use_gpu=use_gpu,
            verbose=2,
            n_pml=4,
        )

        solver.wakesolve(wakelength=wakelength, save_J=False)
        os.remove("tests/014_Ez.h5")

    def test_long_impedance(self):
        global wake
        tol = dict(rtol=50 * 1e-5, atol=50 * 1e-5)
        print(np.abs(wake.Z)[::20])
        assert np.allclose(np.abs(wake.Z)[::20], self.Zabs, **tol), (
            "Abs Impedance samples failed"
        )
