# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import abc
from tabnanny import verbose
import time

import h5py
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, hstack, vstack
from scipy.sparse.linalg import cg

from .boundaries import BCsMixin
from .field import Field
from .logger import Logger
from .materials import material_lib
from .plotting import PlotMixin
from .routines import RoutinesMixin

try:
    import cupy as cp
    from cupyx.scipy.sparse import csc_matrix as gpu_sparse_mat
    from cupyx.scipy.sparse.linalg import cg as gpu_cg

    imported_cupyx = True
except ImportError:
    imported_cupyx = False

try:
    from sparse_dot_mkl import csr_matrix as mkl_sparse_mat
    from sparse_dot_mkl import dot_product_mkl

    imported_mkl = True
except ImportError:
    imported_mkl = False


class SolverFIT3D(PlotMixin, RoutinesMixin, BCsMixin):
    def __init__(
        self,
        grid,
        wake=None,
        cfln=0.5,
        dt=None,
        bc_low=["Periodic", "Periodic", "Periodic"],
        bc_high=["Periodic", "Periodic", "Periodic"],
        use_stl=False,
        use_conductors=False,
        use_gpu=False,
        use_mpi=False,
        dtype=np.float64,
        n_pml=12,
        bg=[1.0, 1.0],
        verbose=1,
        kappa_max=5,
        alpha_max=0.1,
        sigma_factor = 1,
        pml_exp = 3,
        cleaning = None,
    ):
        """
        3D time-domain electromagnetic solver based on the Finite Integration
        Technique (FIT).

        Handles mesh and geometry, material assignment, boundary conditions and
        time-stepping. Supports CPU, optional GPU acceleration (cupyx) and MPI
        domain decomposition. Provides utilities for importing conductors and
        STL solids, applying PML/ABC boundaries, and saving/restoring solver
        state.

        Parameters
        ----------
        grid : GridFIT3D
            Instance providing mesh, coordinate arrays and geometry flags.
        wake : WakeSolver, optional
            Wakefield object with beam parameters used for wake computations.
        cfln : float, optional
            CFL number used to compute a stable timestep when ``dt`` is None.
        dt : float, optional
            Explicit timestep. If provided, it overrides the CFL-based value.
        bc_low, bc_high : list of str, optional
            Boundary conditions for low/high faces in (x, y, z) order.
        use_stl : bool, optional
            If True, apply solids and materials provided in the ``grid`` object.
        use_conductors : bool, optional
            If True, import conductor geometry from ``conductors.py`` masks.
        use_gpu : bool, optional
            Enable GPU acceleration via ``cupyx`` (if available).
        use_mpi : bool, optional
            Enable MPI execution for a subdivided grid.
        dtype : numpy dtype, optional
            Numeric dtype for solver arrays (default ``np.float64``).
        n_pml : int, optional
            Number of PML cells for PML boundary regions.
        bg : sequence or str, optional
            Background material [eps_r, mu_r, sigma] or a material key from
            the library. If a sigma value is provided conductivity handling is
            enabled.
        verbose : int or bool, optional
            Verbosity flag for initialization messages.

        Attributes
        ----------
        E, H, J : wakis.Field
            Electric field, magnetic field and current density containers.
            Access components via labels 'x','y','z'. Example:
            ``solver.E[:, :, n, 'z']`` gives Ez at z-index n.
        ieps, imu, sigma : wakis.Field
            Material tensors (inverse permittivity, inverse permeability and
            conductivity) stored per field component.
        grid : GridFIT3D
            Reference to the input grid object.
        dt : float
            Time-step used for time integration.
        cfln : float
            CFL number used when computing dt from grid spacing.
        """

        self.verbose = verbose
        t0 = time.time()
        self.logger = Logger()

        # Flags
        self.step_0 = True
        self.nstep = int(0)
        self.plotter_active = False
        self.use_conductors = use_conductors
        self.use_stl = use_stl
        self.use_gpu = use_gpu
        self.use_mpi = use_mpi
        self.activate_abc = False  # Will turn true if abc BCs are chosen
        self.activate_pml = False  # Will turn true if pml BCs are chosen
        self.cleaning = cleaning
        self.use_conductivity = (
            False  # Will turn true with conductive material or pml
        )
        self.imported_mkl = imported_mkl  # Use MKL backend when available
        self.one_step = self._one_step
        if use_stl:
            self.use_conductors = False
        self.update_logger(["use_gpu", "use_mpi"])

        # Grid
        self.grid = grid
        self.background = bg
        self.Nx = self.grid.Nx
        self.Ny = self.grid.Ny
        self.Nz = self.grid.Nz
        self.N = self.Nx * self.Ny * self.Nz

        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.dz = self.grid.dz

        self.x = self.grid.x[:-1] + self.dx / 2
        self.y = self.grid.y[:-1] + self.dy / 2
        self.z = self.grid.z[:-1] + self.dz / 2

        self.L = self.grid.L
        self.iL = self.grid.iL
        self.A = self.grid.A
        self.iA = self.grid.iA
        self.tL = self.grid.tL
        self.itL = self.grid.itL
        self.tA = self.grid.tA
        self.itA = self.grid.itA
        self.iV = self.grid.iV
        self.itV = self.grid.itV
        self.update_logger(["grid", "background"])

        # Wake computation
        self.wake = wake
        if self.wake is not None:
            self.logger.wakeSolver = self.wake.logger.wakeSolver

        # Fields
        self.dtype = dtype
        self.E = Field(
            self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype
        )
        self.H = Field(
            self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype
        )
        self.J = Field(
            self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype
        )

        # MPI init
        if self.use_mpi:
            if self.grid.use_mpi:
                self._mpi_initialize()
                self.one_step = self._mpi_one_step
            else:
                print(
                    "[!] Grid not subdivided for MPI, set `use_mpi`=True also in \
                    `GridFIT3D` to enable MPI"
                )

        # Matrices
        if verbose:
            print("Assembling operator matrices...")
        N = self.N
        self.Px = diags([-1, 1], [0, 1], shape=(N, N), dtype=np.int8)
        self.Py = diags([-1, 1], [0, self.Nx], shape=(N, N), dtype=np.int8)
        self.Pz = diags(
            [-1, 1], [0, self.Nx * self.Ny], shape=(N, N), dtype=np.int8
        )

        # original grid
        self.Ds = diags(
            self.L.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype
        )
        self.iDa = diags(
            self.iA.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype
        )

        # tilde grid
        self.tDs = diags(
            self.tL.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype
        )
        self.itDa = diags(
            self.itA.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype
        )

        # Curl matrix
        self.C = vstack(
            [
                hstack([sparse_mat((N, N)), -self.Pz, self.Py]),
                hstack([self.Pz, sparse_mat((N, N)), -self.Px]),
                hstack([-self.Py, self.Px, sparse_mat((N, N))]),
            ],
            dtype=np.int8,
        )

        # Boundaries
        if verbose:
            print("Applying boundary conditions...")
        self.bc_low = bc_low
        self.bc_high = bc_high
        self.update_logger(["bc_low", "bc_high"])
        self._apply_bc_to_C()

        # Materials
        if verbose:
            print("Adding material tensors...")
        if type(bg) is str:
            bg = material_lib[bg.lower()]

        if len(bg) == 3:
            self.eps_bg, self.mu_bg, self.sigma_bg = (
                bg[0] * eps_0,
                bg[1] * mu_0,
                bg[2],
            )
            self.use_conductivity = True
        else:
            self.eps_bg, self.mu_bg, self.sigma_bg = (
                bg[0] * eps_0,
                bg[1] * mu_0,
                0.0,
            )

        # fmt: off
        self.ieps = (
            Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
            * (1.0 / self.eps_bg)
        )
        self.imu = (
            Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
            * (1.0 / self.mu_bg)
        )
        self.sigma = (
            Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
            * self.sigma_bg
        )
        # fmt: on

        if self.use_stl:
            self._apply_stl_materials()

        # Fill PML BCs
        if self.activate_pml:
            if verbose:
                print("Filling PML sigmas...")
            self.one_step = self._one_step_cpml
            self.sigma_factor = sigma_factor
            self.pml_exp = pml_exp
            self.n_pml = n_pml
            self.kappa_max = kappa_max
            self.alpha_max = alpha_max
            self.cleaning = "direct"  # To automatically clean the charge building up at the boundary of the PML
            self._initialize_PML()
            self.update_logger(["n_pml", "kappa_max", "alpha_max", "sigma_factor", "pml_exp"])
            self.one_step = self._one_step_cpml

        # Timestep calculation
        if verbose:
            print("Calculating maximal stable timestep...")
        self.cfln = cfln
        if dt is None:
            self.dt = cfln / (
                c_light
                * np.sqrt(
                    1 / np.min(self.grid.dx) ** 2
                    + 1 / np.min(self.grid.dy) ** 2
                    + 1 / np.min(self.grid.dz) ** 2
                )
            )
        else:
            self.dt = dt
        self.dt = self.dtype(self.dt)

        if self.use_conductivity:  # relaxation time criterion tau
            mask = np.logical_and(
                self.sigma.toarray() != 0,  # for non-conductive
                self.ieps.toarray() != 0,
            )  # for PEC eps=inf

            self.tau = (1 / self.ieps.toarray()[mask]) / self.sigma.toarray()[
                mask
            ]

            if self.dt > self.tau.min():
                self.dt = self.tau.min()
                
        self.update_logger(["dt"])

        # Pre-computing
        if verbose:
            print("Pre-computing...")
        self.iDeps = diags(
            self.ieps.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype
        )
        self.iDmu = diags(
            self.imu.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype
        )
        self.Dsigma = diags(
            self.sigma.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype
        )

        # Matrices for lengths and areas for split field calculations
        if self.activate_pml:

            self.tLx = diags(
                self.tL.field_x, shape=(N, N), dtype=self.dtype
            )
            self.tLy = diags(
                self.tL.field_y, shape=(N, N), dtype=self.dtype
            )
            self.tLz = diags(
                self.tL.field_z, shape=(N, N), dtype=self.dtype
            )
            self.iAx = diags(
                self.iA.field_x, shape=(N, N), dtype=self.dtype
            )
            self.iAy = diags(
                self.iA.field_y, shape=(N, N), dtype=self.dtype
            )
            self.iAz = diags(
                self.iA.field_z, shape=(N, N), dtype=self.dtype
            )
            self.Lx = diags(
                self.L.field_x, shape=(N, N), dtype=self.dtype
            )
            self.Ly = diags(
                self.L.field_y, shape=(N, N), dtype=self.dtype
            )
            self.Lz = diags(
                self.L.field_z, shape=(N, N), dtype=self.dtype
            )
            self.itAx = diags(
                self.itA.field_x, shape=(N, N), dtype=self.dtype
            )
            self.itAy = diags(
                self.itA.field_y, shape=(N, N), dtype=self.dtype
            )
            self.itAz = diags(
                self.itA.field_z, shape=(N, N), dtype=self.dtype
            )
            self.ikapx = diags(
                1.0 / self.kappa.field_x, shape=(N, N), dtype=self.dtype
            )
            self.ikapy = diags(
                1.0 / self.kappa.field_y, shape=(N, N), dtype=self.dtype
            )
            self.ikapz = diags(
                1.0 / self.kappa.field_z, shape=(N, N), dtype=self.dtype
            )

            self.Px = self.ikapx * self.Px
            self.Py = self.ikapy * self.Py
            self.Pz = self.ikapz * self.Pz            

            self.dxy = self.iAx * self.Py * self.Lz * self.Dbc_z
            self.dxz = self.iAx * self.Pz * self.Ly * self.Dbc_y
            self.dyz = self.iAy * self.Pz * self.Lx * self.Dbc_x
            self.dyx = self.iAy * self.Px * self.Lz * self.Dbc_z
            self.dzx = self.iAz * self.Px * self.Ly * self.Dbc_y
            self.dzy = self.iAz * self.Py * self.Lx * self.Dbc_x

            self.dtxy = self.itAx * self.Dbc_z.transpose() * -self.Py.transpose() * self.tLz
            self.dtxz = self.itAx * self.Dbc_y.transpose() * -self.Pz.transpose() * self.tLy
            self.dtyz = self.itAy * self.Dbc_x.transpose() * -self.Pz.transpose() * self.tLx
            self.dtyx = self.itAy * self.Dbc_z.transpose() * -self.Px.transpose() * self.tLz
            self.dtzx = self.itAz * self.Dbc_y.transpose() * -self.Px.transpose() * self.tLy
            self.dtzy = self.itAz * self.Dbc_x.transpose() * -self.Py.transpose() * self.tLx

            self.pml_b = (
            Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
            )
            self.pml_c = (
            Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
            )

            # Convolution Parameter computation, only valid if sigma is zero in physical domain
            oneField = np.ones(self.N * 3, dtype=self.dtype)
            self.pml_b.fromarray(np.exp(
                -(self.sigma.toarray() / (self.kappa.toarray()*eps_0) + self.alpha.toarray()/ eps_0) * self.dt))
            denom = self.sigma.toarray() + self.kappa.toarray() * self.alpha.toarray()
            ratio = np.divide(self.sigma.toarray(), denom, out=np.zeros_like(self.sigma.toarray()), where=denom != 0)
            self.pml_c.fromarray(ratio * (self.pml_b.toarray() - oneField))   

            self.psiHa = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype)
            self.psiHb = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype)
            self.psiEa = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype)
            self.psiEb = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype)

            # Stretched coordinates PML
            self.C = vstack(
                [
                    hstack([sparse_mat((N, N)), -self.Pz, self.Py]),
                    hstack([self.Pz, sparse_mat((N, N)), -self.Px]),
                    hstack([-self.Py, self.Px, sparse_mat((N, N))]),
                ],
                dtype=np.float64,
            )

        if self.cleaning is not None:
            print("Initializing divergence cleaning operators...")
            self.iDv = diags(self.iV, shape=(N, N), dtype=self.dtype)
            self.itDv = diags(self.itV, shape=(N, N), dtype=self.dtype)
            self.Da = diags(self.A.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)
            self.tDa = diags(self.tA.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)
            self.iDs = diags(self.iL.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)
            self.itDs = diags(self.itL.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)
            self.Deps = diags(1.0 / self.ieps.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)

            # Grading for direct cleaning, can be used to ramp up the cleaning strength towards the boundaries to avoid reflections from the cleaning itself, especially when using few cleaning cells. This is not needed for Poisson cleaning which is more stable and less reflective, but can help to improve stability for direct cleaning.

            self.cleaning_mask = Field(self.Nx, self.Ny, self.Nz, use_ones=False, dtype=self.dtype, use_gpu=self.use_gpu)
            self.damp = (Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype, use_gpu=self.use_gpu))
            cleanStart = 0
            if self.activate_pml:
                cleanStart = self.n_pml

            for d in ['x', 'y', 'z']:
                self.cleaning_mask[:, :, :cleanStart+3, d] = 1.0
                self.cleaning_mask[:, :, -cleanStart-4:, d] = 1.0
                self.damp[:, :, :cleanStart+1, d] = 0
                self.damp[:, :, -cleanStart-2:, d] = 0
            
            # Topological operators
            self.S = hstack([self.Px, self.Py, self.Pz], dtype=self.dtype) * self.Dbc
            self.tS = hstack([self.Px.transpose(), self.Py.transpose(), self.Pz.transpose()], dtype=self.dtype) * self.Dbc.transpose()

            # Cleaning operators
            self.Div = (self.itDv @ self.tS @ self.tDa)
            self.Grad = (self.iDs @ self.tS.transpose())
            self.Lag = self.Div @ self.Grad

            #Plot variables
            self.correct = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype)

            if self.use_gpu:
                self.phi = cp.zeros(self.N, dtype=self.dtype)
                self.rho = cp.zeros(self.N, dtype=self.dtype)
                self.rhoplot = cp.zeros(self.N, dtype=self.dtype)

            else:
                self.phi = np.zeros(self.N, dtype=self.dtype)
                self.rho = np.zeros(self.N, dtype=self.dtype)


        self.tDsiDmuiDaC = self.iDa * self.iDmu * self.C * self.Ds
        self.itDaiDepsDstC = (
            self.iDeps * self.itDa * self.C.transpose() * self.tDs
        )

        if imported_mkl and not self.use_gpu:  # MKL backend for CPU
            if verbose:
                print("Using MKL backend for time-stepping...")
            self.tDsiDmuiDaC = mkl_sparse_mat(self.tDsiDmuiDaC)
            self.itDaiDepsDstC = mkl_sparse_mat(self.itDaiDepsDstC)
            self.one_step = (
                self._mpi_one_step_mkl if self.use_mpi else self._one_step_mkl
            )
            if self.activate_pml:
                self.dxy = mkl_sparse_mat(self.dxy)
                self.dxz = mkl_sparse_mat(self.dxz)
                self.dyz = mkl_sparse_mat(self.dyz)
                self.dyx = mkl_sparse_mat(self.dyx)
                self.dzx = mkl_sparse_mat(self.dzx)
                self.dzy = mkl_sparse_mat(self.dzy)
                self.dtxy = mkl_sparse_mat(self.dtxy)
                self.dtxz = mkl_sparse_mat(self.dtxz)
                self.dtyz = mkl_sparse_mat(self.dtyz)
                self.dtyx = mkl_sparse_mat(self.dtyx)
                self.dtzx = mkl_sparse_mat(self.dtzx)
                self.dtzy = mkl_sparse_mat(self.dtzy)

        # Move to GPU
        if use_gpu:
            if verbose:
                print("Moving to GPU...")
            if imported_cupyx:
                self.tDsiDmuiDaC = gpu_sparse_mat(self.tDsiDmuiDaC)
                self.itDaiDepsDstC = gpu_sparse_mat(self.itDaiDepsDstC)
                self.ieps.to_gpu()
                self.sigma.to_gpu()
                self.imu.to_gpu()

                if self.activate_pml:
                    self.dxy = gpu_sparse_mat(self.dxy)
                    self.dxz = gpu_sparse_mat(self.dxz)
                    self.dyz = gpu_sparse_mat(self.dyz)
                    self.dyx = gpu_sparse_mat(self.dyx)
                    self.dzx = gpu_sparse_mat(self.dzx)
                    self.dzy = gpu_sparse_mat(self.dzy)
                    self.dtxy = gpu_sparse_mat(self.dtxy)
                    self.dtxz = gpu_sparse_mat(self.dtxz)
                    self.dtyz = gpu_sparse_mat(self.dtyz)
                    self.dtyx = gpu_sparse_mat(self.dtyx)
                    self.dtzx = gpu_sparse_mat(self.dtzx)
                    self.dtzy = gpu_sparse_mat(self.dtzy)
                    self.pml_b.to_gpu()
                    self.pml_c.to_gpu()

                if self.cleaning is not None:
                    self.Div = gpu_sparse_mat(self.Div)
                    self.Grad = gpu_sparse_mat(self.Grad)
                    self.Lag = gpu_sparse_mat(self.Lag)
                    self.Deps = gpu_sparse_mat(self.Deps)
                    self.iDeps = gpu_sparse_mat(self.iDeps)
                    
            else:
                raise ImportError(
                    "[!] cupyx could not be imported, please check CUDA installation"
                )

        if verbose:
            print(f"Total solver initialization time: {time.time() - t0} s")

        self.solverInitializationTime = time.time() - t0
        self.update_logger(["solverInitializationTime"])

    def update_tensors(self, tensor="all"):
        """
        Update tensor matrices after material Field changes and precompute
        combined operators used for time-stepping.

        When ``ieps``, ``imu`` or ``sigma`` are modified this routine
        reconstructs the corresponding sparse diagonal matrices and the
        composite operator products used in the update equations. Use the
        ``tensor`` argument to restrict work to a single tensor for efficiency.

        Parameters
        ----------
        tensor : {'ieps','imu','sigma','all'}, optional
            Which tensor to update. Default is 'all' which recomputes every
            tensor and refreshes the precomputed time-stepping matrices.
        """
        if self.verbose:
            print(f'Re-computing tensor "{tensor}"...')

        if tensor == "ieps":
            self.iDeps = diags(
                self.ieps.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
        elif tensor == "imu":
            self.iDmu = diags(
                self.imu.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
        elif tensor == "sigma":
            self.Dsigma = diags(
                self.sigma.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
        elif tensor == "all":
            self.iDeps = diags(
                self.ieps.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.iDmu = diags(
                self.imu.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.Dsigma = diags(
                self.sigma.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )

        if self.verbose:
            print("Re-Pre-computing ...")
        self.tDsiDmuiDaC = self.iDa * self.iDmu * self.C * self.Ds
        self.itDaiDepsDstC = (
            self.iDeps * self.itDa * self.C.transpose() * self.tDs
        )
        self.step_0 = False

    def apply_cleaning(self):        
        self.J.fromarray(self.J.toarray() * self.damp.toarray()) # To addditionally Damp J
        self.rho -= self.dt * (self.Div @ (self.J.toarray() * self.J_mask)) # Apply cleaning mask to rho update to clean only specific parts of the domain

        if (self.J_mask * self.cleaning_mask.toarray()).max() == 0: # clean only if J intersects enough with the cleaning mask
            return
            
        self.rho = self.rho * self.cleaning_mask.field_x # Apply cleaning mask to rho to clean only specific parts of the domain
        if verbose > 1:
            print("Apply cleaning")
        if self.use_gpu:
            self.phi, info = gpu_cg(self.Lag, self.rho, x0=self.phi, maxiter=1000)
        else:
            self.phi, info = cg(self.Lag, self.rho, x0=self.phi, maxiter=1000)
        if info != 0:
            print(f'[!] Poisson solver did not converge, info: {info}')
        self.correct.fromarray(self.iDeps * (self.Grad @ self.phi))
        self.E.fromarray(self.E.toarray() - self.correct.toarray())
        if verbose > 1:
            self.rhoplot = self.rho.copy() # For plotting purposes
        self.rho = np.zeros_like(self.rho)

    def _one_step_cpml(self):
    # Including the convolutional terms for the CPML update equations
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()
            self.J_old = np.zeros_like(self.J.toarray())
            if self.verbose>1:
                    print("Starting time-stepping with CPML...")
    
        dxyEz = self.dxy * self.E.field_z
        dxzEy = self.dxz * self.E.field_y
        dyxEz = self.dyx * self.E.field_z
        dyzEx = self.dyz * self.E.field_x
        dzxEy = self.dzx * self.E.field_y
        dzyEx = self.dzy * self.E.field_x
        
        self.psiHa.field_x = self.pml_b.field_y * self.psiHa.field_x + self.pml_c.field_y * dxyEz
        self.psiHb.field_x = self.pml_b.field_z * self.psiHb.field_x + self.pml_c.field_z * dxzEy
        self.psiHb.field_y = self.pml_b.field_x * self.psiHb.field_y + self.pml_c.field_x * dyxEz
        self.psiHa.field_y = self.pml_b.field_z * self.psiHa.field_y + self.pml_c.field_z * dyzEx
        self.psiHa.field_z = self.pml_b.field_x * self.psiHa.field_z + self.pml_c.field_x * dzxEy
        self.psiHb.field_z = self.pml_b.field_y * self.psiHb.field_z + self.pml_c.field_y * dzyEx

        self.H.field_x = (self.H.field_x - self.dt * self.imu.field_x * (dxyEz - dxzEy)
        ) - self.dt * self.imu.field_x * (self.psiHa.field_x - self.psiHb.field_x)
        self.H.field_y = (self.H.field_y - self.dt * self.imu.field_y * (dyzEx - dyxEz)
        ) - self.dt * self.imu.field_y * (self.psiHa.field_y - self.psiHb.field_y)           
        self.H.field_z = (self.H.field_z - self.dt * self.imu.field_z * (dzxEy - dzyEx)
        ) - self.dt * self.imu.field_z * (self.psiHa.field_z - self.psiHb.field_z)

        dtxyHz = self.dtxy * self.H.field_z
        dtxzHy = self.dtxz * self.H.field_y
        dtyxHz = self.dtyx * self.H.field_z
        dtyzHx = self.dtyz * self.H.field_x
        dtzxHy = self.dtzx * self.H.field_y
        dtzyHx = self.dtzy * self.H.field_x

        Jtemp = self.sigma.toarray() * self.E.toarray()
        dJ = (Jtemp - self.J_old)
        self.J.fromarray(self.J.toarray() + dJ)
        self.J_old = Jtemp

        if self.cleaning is not None:
            self.apply_cleaning()

        self.psiEa.field_x = self.pml_b.field_y * self.psiEa.field_x + self.pml_c.field_y * dtxyHz
        self.psiEb.field_x = self.pml_b.field_z * self.psiEb.field_x + self.pml_c.field_z * dtxzHy
        self.psiEb.field_y = self.pml_b.field_x * self.psiEb.field_y + self.pml_c.field_x * dtyxHz
        self.psiEa.field_y = self.pml_b.field_z * self.psiEa.field_y + self.pml_c.field_z * dtyzHx
        self.psiEa.field_z = self.pml_b.field_x * self.psiEa.field_z + self.pml_c.field_x * dtzxHy
        self.psiEb.field_z = self.pml_b.field_y * self.psiEb.field_z + self.pml_c.field_y * dtzyHx

        self.E.field_x = (self.E.field_x + self.dt * self.ieps.field_x * (dtxyHz - dtxzHy)
                            - self.dt * self.ieps.field_x * self.J.field_x
                            + self.dt * self.ieps.field_x * (self.psiEa.field_x - self.psiEb.field_x))
        self.E.field_y = (self.E.field_y + self.dt * self.ieps.field_y * (dtyzHx - dtyxHz) 
                            - self.dt * self.ieps.field_y * self.J.field_y
                            + self.dt * self.ieps.field_y * (self.psiEa.field_y - self.psiEb.field_y))            
        self.E.field_z = (self.E.field_z + self.dt * self.ieps.field_z * (dtzxHy - dtzyHx) 
                            - self.dt * self.ieps.field_z * self.J.field_z
                            + self.dt * self.ieps.field_z * (self.psiEa.field_z - self.psiEb.field_z))

    def _one_step(self):
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()
            self.J_old = np.zeros_like(self.J.toarray())
        self.H.fromarray(
            self.H.toarray() - self.dt * self.tDsiDmuiDaC * self.E.toarray()
        )

        # include current computation
        if self.use_conductivity:
            Jtemp = self.sigma.toarray() * self.E.toarray()
            dJ = (Jtemp - self.J_old)
            self.J.fromarray(self.J.toarray() + dJ)
            self.J_old = Jtemp

        if self.cleaning is not None:
            self.apply_cleaning()
        
        self.E.fromarray(
            self.E.toarray()
            + self.dt
            * (
                self.itDaiDepsDstC * self.H.toarray()
                - self.ieps.toarray() * self.J.toarray()
            )
        )

    def _one_step_mkl(self):
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()
            self.J_old = np.zeros_like(self.J.toarray())
        if self.activate_pml:


            dxyEz = dot_product_mkl(self.dxy, self.E.field_z)
            dxzEy = dot_product_mkl(self.dxz, self.E.field_y)
            dyxEz = dot_product_mkl(self.dyx, self.E.field_z)
            dyzEx = dot_product_mkl(self.dyz, self.E.field_x)
            dzxEy = dot_product_mkl(self.dzx, self.E.field_y)
            dzyEx = dot_product_mkl(self.dzy, self.E.field_x)
            
            self.psiHa.field_x = self.pml_b.field_x * self.psiHa.field_x + self.pml_c.field_x * dxyEz
            self.psiHb.field_x = self.pml_b.field_z * self.psiHb.field_x + self.pml_c.field_z * dxzEy
            self.psiHb.field_y = self.pml_b.field_x * self.psiHb.field_y + self.pml_c.field_x * dyxEz
            self.psiHa.field_y = self.pml_b.field_z * self.psiHa.field_y + self.pml_c.field_z * dyzEx
            self.psiHa.field_z = self.pml_b.field_x * self.psiHa.field_z + self.pml_c.field_x * dzxEy
            self.psiHb.field_z = self.pml_b.field_y * self.psiHb.field_z + self.pml_c.field_y * dzyEx

            self.H.field_x = (self.H.field_x - self.dt * self.imu.field_x * (dxyEz - dxzEy)
            ) - self.dt * self.imu.field_x * (self.psiHa.field_x - self.psiHb.field_x)
            self.H.field_y = (self.H.field_y - self.dt * self.imu.field_y * (dyzEx - dyxEz)
            ) - self.dt * self.imu.field_y * (self.psiHa.field_y - self.psiHb.field_y)           
            self.H.field_z = (self.H.field_z - self.dt * self.imu.field_z * (dzxEy - dzyEx)
            ) - self.dt * self.imu.field_z * (self.psiHa.field_z - self.psiHb.field_z)

            dtxyHz = dot_product_mkl(self.dtxy, self.H.field_z)
            dtxzHy = dot_product_mkl(self.dtxz, self.H.field_y)
            dtyxHz = dot_product_mkl(self.dtyx, self.H.field_z)
            dtyzHx = dot_product_mkl(self.dtyz, self.H.field_x)
            dtzxHy = dot_product_mkl(self.dtzx, self.H.field_y)
            dtzyHx = dot_product_mkl(self.dtzy, self.H.field_x)

            self.psiEa.field_x = self.pml_b.field_y * self.psiEa.field_x + self.pml_c.field_y * dtxyHz
            self.psiEb.field_x = self.pml_b.field_z * self.psiEb.field_x + self.pml_c.field_z * dtxzHy
            self.psiEb.field_y = self.pml_b.field_x * self.psiEb.field_y + self.pml_c.field_x * dtyxHz
            self.psiEa.field_y = self.pml_b.field_z * self.psiEa.field_y + self.pml_c.field_z * dtyzHx
            self.psiEa.field_z = self.pml_b.field_x * self.psiEa.field_z + self.pml_c.field_x * dtzxHy
            self.psiEb.field_z = self.pml_b.field_y * self.psiEb.field_z + self.pml_c.field_y * dtzyHx

            Jtemp = self.sigma.toarray() * self.E.toarray()
            self.dJ = (Jtemp - self.J_old)
            self.J.fromarray(self.J.toarray() + self.dJ)
            self.J_old = Jtemp

            if self.cleaning is not None:
                self.apply_cleaning()

            self.E.field_x = (self.E.field_x + self.dt * self.ieps.field_x * (dtxyHz - dtxzHy)
                                - self.dt * self.ieps.field_x * self.J.field_x
                                + self.dt * self.ieps.field_x * (self.psiEa.field_x - self.psiEb.field_x))
            self.E.field_y = (self.E.field_y + self.dt * self.ieps.field_y * (dtyzHx - dtyxHz) 
                                - self.dt * self.ieps.field_y * self.J.field_y
                                + self.dt * self.ieps.field_y * (self.psiEa.field_y - self.psiEb.field_y))            
            self.E.field_z = (self.E.field_z + self.dt * self.ieps.field_z * (dtzxHy - dtzyHx) 
                                - self.dt * self.ieps.field_z * self.J.field_z
                                + self.dt * self.ieps.field_z * (self.psiEa.field_z - self.psiEb.field_z))  

        else:       
            self.H.fromarray(
                self.H.toarray()
                - self.dt * dot_product_mkl(self.tDsiDmuiDaC, self.E.toarray())
            )

            if self.use_conductivity:
                Jtemp = self.sigma.toarray() * self.E.toarray()# * self.damp.toarray()
                self.dJ = (Jtemp - self.J_old)
                self.J.fromarray(self.J.toarray() + self.dJ)
                self.J_old = Jtemp

            if self.cleaning is not None:
                self.apply_cleaning()

            self.E.fromarray(
                self.E.toarray()
                + self.dt
                * (
                    dot_product_mkl(self.itDaiDepsDstC, self.H.toarray())
                    - self.ieps.toarray() * self.J.toarray()
                )
            )

    def _mpi_initialize(self):
        self.comm = self.grid.comm
        self.rank = self.grid.rank
        self.size = self.grid.size

        self.NZ = self.grid.NZ
        self.ZMIN = self.grid.ZMIN
        self.ZMAX = self.grid.ZMAX
        self.Z = self.grid.Z

    def _mpi_one_step(self):
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()
            self.J_old = np.zeros_like(self.J.toarray())

        self.H.fromarray(
            self.H.toarray() - self.dt * self.tDsiDmuiDaC * self.E.toarray()
        )

        self._mpi_communicate(self.H)
        self._mpi_communicate(self.J)
        self.E.fromarray(
            self.E.toarray()
            + self.dt
            * (
                self.itDaiDepsDstC * self.H.toarray()
                - self.ieps.toarray() * self.J.toarray()
            )
        )

        self._mpi_communicate(self.E)
        # include current computation
        if self.use_conductivity:
            Jtemp = self.sigma.toarray() * self.E.toarray()
            dJ = (Jtemp - self.J_old)
            self.J.fromarray(self.J.toarray() + dJ)
            self.J_old = Jtemp

    def _mpi_one_step_mkl(self):
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()
            self.J_old = np.zeros_like(self.J.toarray())
        self.H.fromarray(
            self.H.toarray()
            - self.dt * dot_product_mkl(self.tDsiDmuiDaC, self.E.toarray())
        )

        self._mpi_communicate(self.H)
        self._mpi_communicate(self.J)

        self.E.fromarray(
            self.E.toarray()
            + self.dt
            * (
                dot_product_mkl(self.itDaiDepsDstC, self.H.toarray())
                - self.ieps.toarray() * self.J.toarray()
            )
        )

        self._mpi_communicate(self.E)
        # include current computation
        if self.use_conductivity:
            Jtemp = self.sigma.toarray() * self.E.toarray()
            dJ = (Jtemp - self.J_old)
            self.J.fromarray(self.J.toarray() + dJ)
            self.J_old = Jtemp

    def _mpi_communicate(self, field):
        if self.use_gpu:
            field.from_gpu()

        # ghosts lo
        if self.rank > 0:
            for d in ["x", "y", "z"]:
                self.comm.Sendrecv(
                    field[:, :, 1, d],
                    recvbuf=field[:, :, 0, d],
                    dest=self.rank - 1,
                    sendtag=0,
                    source=self.rank - 1,
                    recvtag=1,
                )
        # ghosts hi
        if self.rank < self.size - 1:
            for d in ["x", "y", "z"]:
                self.comm.Sendrecv(
                    field[:, :, -2, d],
                    recvbuf=field[:, :, -1, d],
                    dest=self.rank + 1,
                    sendtag=1,
                    source=self.rank + 1,
                    recvtag=0,
                )

        if self.use_gpu:
            field.to_gpu()

    def mpi_gather(self, field, x=None, y=None, z=None, component=None):
        """
        Gather a component or slice of a distributed Field from all MPI ranks.

        Assumes the field is split along the z-axis among ranks. The function
        collects local buffers, removes ghost cells and concatenates rank
        contributions to build a global NumPy array on the root rank (rank 0).

        Parameters
        ----------
        field : str or wakis.Field
            Field identifier ('E','H','J') optionally with a component suffix
            (e.g. 'Ex'), or a ``wakis.Field`` object. If no component is given
            the 'z' component is used by default.
        x, y, z : int or slice, optional
            Index or slice for each axis to gather. Defaults to the full range.
        component : {'x','y','z'} or slice, optional
            Component to gather when ``field`` is a Field object.

        Returns
        -------
        numpy.ndarray or None
            Assembled global array on rank 0; returns ``None`` on non-root ranks.
        """

        if x is None:
            x = slice(0, self.Nx)
        if y is None:
            y = slice(0, self.Ny)
        if z is None:
            z = slice(0, self.NZ)

        if type(field) is str:
            if len(field) == 2:  # support for e.g. field='Ex'
                component = field[1]
                field = field[0]
            elif len(field) == 4:  # support for Abs
                component = field[1:]
                field = field[0]
            elif component is None:
                component = "z"
                print(
                    "[!] `component` not specified, using default component='z'"
                )

            if field == "E":
                local = self.E[x, y, :, component].ravel()
            elif field == "H":
                local = self.H[x, y, :, component].ravel()
            elif field == "J":
                local = self.J[x, y, :, component].ravel()
        else:
            if component is None:
                component = "z"
                print(
                    "[!] `component` not specified, using default component='z'"
                )
            local = field[x, y, :, component].ravel()

        buffer = self.comm.gather(local, root=0)
        _field = None

        if self.rank == 0:
            if type(x) is int and type(y) is int:  # 1d array at x=a, y=b
                nz = self.NZ // self.size
                _field = np.zeros((self.NZ))
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[zz] = np.reshape(
                            buffer[r], (nz + self.grid.n_ghosts)
                        )[:-1]
                    elif r == (self.size - 1):
                        _field[zz] = np.reshape(
                            buffer[r], (nz + self.grid.n_ghosts)
                        )[1:]
                    else:
                        _field[zz] = np.reshape(
                            buffer[r], (nz + 2 * self.grid.n_ghosts)
                        )[1:-1]
                _field = _field[z]

            elif type(x) is int:  # 2d slice at x=a
                ny = y.stop - y.start
                nz = self.NZ // self.size
                _field = np.zeros((ny, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[:, zz] = np.reshape(
                            buffer[r], (ny, nz + self.grid.n_ghosts)
                        )[:, :-1]
                    elif r == (self.size - 1):
                        _field[:, zz] = np.reshape(
                            buffer[r], (ny, nz + self.grid.n_ghosts)
                        )[:, 1:]
                    else:
                        _field[:, zz] = np.reshape(
                            buffer[r], (ny, nz + 2 * self.grid.n_ghosts)
                        )[:, 1:-1]
                _field = _field[:, z]

            elif type(y) is int:  # 2d slice at y=a
                nx = x.stop - x.start
                nz = self.NZ // self.size
                _field = np.zeros((nx, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[:, zz] = np.reshape(
                            buffer[r], (nx, nz + self.grid.n_ghosts)
                        )[:, :-1]
                    elif r == (self.size - 1):
                        _field[:, zz] = np.reshape(
                            buffer[r], (nx, nz + self.grid.n_ghosts)
                        )[:, 1:]
                    else:
                        _field[:, zz] = np.reshape(
                            buffer[r], (nx, nz + 2 * self.grid.n_ghosts)
                        )[:, 1:-1]
                _field = _field[:, z]

            else:  # both type slice -> 3d array
                nx = x.stop - x.start
                ny = y.stop - y.start
                nz = self.NZ // self.size
                _field = np.zeros((nx, ny, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[:, :, zz] = np.reshape(
                            buffer[r], (nx, ny, nz + self.grid.n_ghosts)
                        )[:, :, :-1]
                    elif r == (self.size - 1):
                        _field[:, :, zz] = np.reshape(
                            buffer[r], (nx, ny, nz + self.grid.n_ghosts)
                        )[:, :, 1:]
                    else:
                        _field[:, :, zz] = np.reshape(
                            buffer[r], (nx, ny, nz + 2 * self.grid.n_ghosts)
                        )[:, :, 1:-1]
                _field = _field[:, :, z]

        return _field

    def mpi_gather_asField(self, field):
        """
        Gather distributed field data from MPI ranks and return a global Field.

        Collects the full 3-component field (E, H or J) from each rank and
        reconstructs a single ``wakis.Field`` on the root rank. Ghost cells are
        removed when reassembling the per-rank buffers.

        Parameters
        ----------
        field : str or wakis.Field
            Identifier ('E','H','J') or a Field-like object to gather.

        Returns
        -------
        wakis.Field or None
            Global Field object assembled on rank 0. Returns ``None`` on other
            ranks.
        """

        _field = Field(self.Nx, self.Ny, self.NZ)

        for d in ["x", "y", "z"]:
            if type(field) is str:
                if field == "E":
                    local = self.E[:, :, :, d].ravel()
                elif field == "H":
                    local = self.H[:, :, :, d].ravel()
                elif field == "J":
                    local = self.J[:, :, :, d].ravel()
            else:
                local = field[:, :, :, d].ravel()

            buffer = self.comm.gather(local, root=0)
            if self.rank == 0:
                nz = self.NZ // self.size
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[:, :, zz, d] = np.reshape(
                            buffer[r],
                            (self.Nx, self.Ny, nz + self.grid.n_ghosts),
                        )[:, :, :-1]
                    elif r == (self.size - 1):
                        _field[:, :, zz, d] = np.reshape(
                            buffer[r],
                            (self.Nx, self.Ny, nz + self.grid.n_ghosts),
                        )[:, :, 1:]
                    else:
                        _field[:, :, zz, d] = np.reshape(
                            buffer[r],
                            (self.Nx, self.Ny, nz + 2 * self.grid.n_ghosts),
                        )[:, :, 1:-1]

        return _field

    def _set_ghosts_to_0(self):
        """
        Zero-out ghost-cell field values used for MPI and boundary exchange.

        Clears any initial condition values that were accidentally placed in
        ghost cells so that subsequent MPI sends/receives and boundary updates
        behave correctly.
        """
        # Set H ghost quantities to 0
        for d in ["x", "y", "z"]:  # tangential to zero
            if d != "x":
                self.H[-1, :, :, d] = 0.0
            if d != "y":
                self.H[:, -1, :, d] = 0.0
            if d != "z":
                self.H[:, :, -1, d] = 0.0

        # Set E ghost quantities to 0
        self.E[-1, :, :, "x"] = 0.0
        self.E[:, -1, :, "y"] = 0.0
        self.E[:, :, -1, "z"] = 0.0

    def _apply_conductors(self):
        """
        Apply PEC conductor masking by zeroing inverse-permittivity inside
        conductor volumes.

        This enforces tangential electric field cancellation inside conductor
        regions by setting the local 1/epsilon to zero.
        """
        self.flag_in_conductors = (
            self.grid.flag_int_cell_yz[:-1, :, :]
            + self.grid.flag_int_cell_zx[:, :-1, :]
            + self.grid.flag_int_cell_xy[:, :, :-1]
        )

        self.ieps *= self.flag_in_conductors

    def _set_field_in_conductors_to_0(self):
        """
        Zero dynamic fields inside conductor masks.

        Ensures that any initial E/H fields mapped into conductor regions are
        removed before time-stepping, avoiding non-physical behaviour.
        """
        self.flag_cleanup = (
            self.grid.flag_int_cell_yz[:-1, :, :]
            + self.grid.flag_int_cell_zx[:, :-1, :]
            + self.grid.flag_int_cell_xy[:, :, :-1]
        )

        self.H *= self.flag_cleanup
        self.E *= self.flag_cleanup

    def _apply_stl_materials(self):
        """
        Mask STL solids in the grid and assign user-defined materials.

        Iterates over STL solids imported in the grid and updates ``ieps``,
        ``imu`` and ``sigma`` according to the material provided for each
        solid. Materials may be referenced by a library key (string) or given
        as explicit tuples (eps_r, mu_r[, sigma]). Inverse permittivity and
        inverse permeability values are stored in the corresponding Fields.

        Notes
        -----
        - STL material values must be relative (eps_r, mu_r).
        - Supply conductivity explicitly to enable conductive behaviour.
        """
        grid = self.grid.grid
        self.stl_solids = self.grid.stl_solids
        self.stl_materials = self.grid.stl_materials
        self.stl_colors = self.grid.stl_colors

        for key in self.stl_solids.keys():
            mask = np.reshape(grid[key], (self.Nx, self.Ny, self.Nz)).astype(
                int
            )

            if type(self.stl_materials[key]) is str:
                # Retrieve from material library
                mat_key = self.stl_materials[key].lower()

                eps = material_lib[mat_key][0] * eps_0
                mu = material_lib[mat_key][1] * mu_0

                # Setting to zero
                self.ieps += self.ieps * (-1.0 * mask)
                self.imu += self.imu * (-1.0 * mask)

                # Adding new values
                self.ieps += mask * 1.0 / eps
                self.imu += mask * 1.0 / mu

                # Conductivity
                if len(material_lib[mat_key]) == 3:
                    sigma = material_lib[mat_key][2]
                    self.sigma += self.sigma * (-1.0 * mask)
                    self.sigma += mask * sigma
                    self.use_conductivity = True

                elif self.sigma_bg > 0.0:  # assumed sigma = 0
                    self.sigma += self.sigma * (-1.0 * mask)

            else:
                # From input
                eps = self.stl_materials[key][0] * eps_0
                mu = self.stl_materials[key][1] * mu_0

                # Setting to zero
                self.ieps += self.ieps * (-1.0 * mask)
                self.imu += self.imu * (-1.0 * mask)

                # Adding new values
                self.ieps += mask * 1.0 / eps
                self.imu += mask * 1.0 / mu

                # Conductivity
                if len(self.stl_materials[key]) == 3:
                    sigma = self.stl_materials[key][2]
                    self.sigma += self.sigma * (-1.0 * mask)
                    self.sigma += mask * sigma
                    self.use_conductivity = True

                elif self.sigma_bg > 0.0:  # assumed sigma = 0
                    self.sigma += self.sigma * (-1.0 * mask)

    def _attrcleanup(self):
        # Fields
        if hasattr(self, "BC"):
            del self.BC
            del self.Dbc
        if self.activate_pml:
           del self.alpha_mask
           #del self.kappa
           #del self.alpha
        del self.L, self.tL, self.iA, self.itA

        # Matrices
        del self.Px, self.Py, self.Pz
        del self.Ds, self.iDa, self.tDs, self.itDa
        del self.C
        if self.activate_pml:
            del self.iAx, self.iAy, self.iAz, self.itAx, self.itAy, self.itAz
            del self.Lx, self.Ly, self.Lz, self.tLx, self.tLy, self.tLz
            del self.ikapx, self.ikapy, self.ikapz

    def save_state(self, filename="solver_state.h5", close=True):
        """
        Save dynamic solver state (H, E, J) to an HDF5 file.

        Writes the core dynamic fields to ``filename``. When running under MPI
        the distributed fields are gathered to the root rank before saving.

        Parameters
        ----------
        filename : str, optional
            Output HDF5 filename. Default is "solver_state.h5".
        close : bool, optional
            If True (default) the file is closed before returning. If False an
            open ``h5py.File`` is returned for caller-managed operations.

        Returns
        -------
        h5py.File or None
            Open file object when ``close`` is False, otherwise None.
        """

        if self.use_mpi:  # MPI savestate
            H = self.mpi_gather_asField("H")
            E = self.mpi_gather_asField("E")
            J = self.mpi_gather_asField("J")

            if self.rank == 0:
                state = h5py.File(filename, "w")
                state.create_dataset("H", data=H)
                state.create_dataset("E", data=E)
                state.create_dataset("J", data=J)
            # TODO: check for MPI-GPU

        elif self.use_gpu:  # GPU savestate
            state = h5py.File(filename, "w")
            state.create_dataset("H", data=self.H.toarray().get())
            state.create_dataset("E", data=self.E.toarray().get())
            state.create_dataset("J", data=self.J.toarray().get())

        else:  # CPU savestate
            state = h5py.File(filename, "w")
            state.create_dataset("H", data=self.H.toarray())
            state.create_dataset("E", data=self.E.toarray())
            state.create_dataset("J", data=self.J.toarray())

        if close:
            state.close()
        else:
            return state  # Caller must close this manually

    def load_state(self, filename="solver_state.h5"):
        """
        Load dynamic solver state (H, E, J) from an HDF5 file and restore them.

        Parameters
        ----------
        filename : str, optional
            Input HDF5 filename. Default is "solver_state.h5".

        Notes
        -----
        Currently performs a simple load from a single-file state. MPI-aware
        redistribution of loaded arrays to worker ranks is TODO.
        """
        state = h5py.File(filename, "r")

        self.E.fromarray(state["E"][:])
        self.H.fromarray(state["H"][:])
        self.J.fromarray(state["J"][:])

        # TODO: support MPI loadstate

        state.close()

    def read_state(self, filename="solver_state.h5"):
        """
        Open an HDF5 file for read-only access without loading its contents.

        Returns an open ``h5py.File`` object that the caller must close when
        finished. This is useful for inspecting saved state without restoring
        it into the solver.

        Parameters
        ----------
        filename : str, optional
            Input HDF5 filename. Default is "solver_state.h5".

        Returns
        -------
        h5py.File
            Open HDF5 file object in read mode.
        """
        return h5py.File(filename, "r")

    def reset_fields(self):
        """
        Reset dynamic field arrays (E, H, J) to zero across the simulation.

        Useful when reusing a ``SolverFIT3D`` instance for a new run without
        reconstructing the entire object.
        """
        for d in ["x", "y", "z"]:
            self.E[:, :, :, d] = 0.0
            self.H[:, :, :, d] = 0.0
            self.J[:, :, :, d] = 0.0

    def update_logger(self, attrs):
        """
        Copy selected solver attributes into the internal ``Logger`` object.

        Parameters
        ----------
        attrs : iterable of str
            Names of attributes to copy to ``self.logger.solver``. Special case
            'grid' copies the grid logger reference instead of a value.
        """
        for atr in attrs:
            if atr == "grid":
                self.logger.grid = self.grid.logger.grid
            else:
                self.logger.solver[atr] = getattr(self, atr)
