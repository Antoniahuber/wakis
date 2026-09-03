import sys

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


@pytest.mark.slow
class TestGridFIT3DMeshing:
    # Reference data
    tol = dict(rtol=50e-5, atol=50e-4)
    dtype = np.float32

    # fmt: off
    WP = np.array([ 3.03999377e-18, -1.57436678e-14, -5.81744067e-11, -4.16962878e-08,
                    -8.37576052e-06, -5.10188367e-04, -9.65024317e-03, -5.62474460e-02,
                    -8.92658495e-02,  1.94104416e-02,  1.20731894e-01,  6.94716656e-02,
                    -2.25460627e-02, -4.91365346e-02, -2.81756714e-02, -1.07040993e-02,
                    6.24067287e-02,  7.76899724e-02, -6.41804617e-02, -1.09100607e-01,
                    3.17122367e-02,  9.95300771e-02,  1.59930539e-02, -6.11959223e-02,
                    -3.38419464e-02, -3.31974531e-03,  1.97809615e-02,  7.32837232e-02,
                    2.14655914e-03, -9.54641911e-02, -3.93029761e-02,  7.85793202e-02,
                    6.92603412e-02, -4.44469131e-02, -5.61724850e-02, -4.31207455e-03,
                    1.37319512e-02,  4.47538136e-02,  3.30728875e-02, -4.91526078e-02,
                    -7.17331871e-02,  2.30389026e-02,  9.07601155e-02,  4.60074950e-03,
                    -6.82039506e-02, -2.42327580e-02,  1.47484248e-02,  3.33755469e-02,
                    3.07827068e-02, -1.10044189e-02, -6.00957767e-02, -2.83398905e-02,
                    7.01179625e-02,  5.07391705e-02, -4.55863327e-02, -5.23725044e-02,
                    3.64085044e-03,  3.55489172e-02,  2.52530595e-02,  3.40678159e-03,
                    -3.39325723e-02, -4.70488102e-02,  2.66807614e-02,  6.77546427e-02,
                    -3.49550941e-03, -5.86094480e-02, -2.46501822e-02,  3.33310938e-02,
                    3.19647879e-02,  3.80802801e-03, -1.52754979e-02, -4.11351141e-02,
                    -7.28641157e-03,  5.23813004e-02,  3.22026092e-02, -3.76132883e-02,
                    -4.92387915e-02,  1.42774059e-02,  4.05807388e-02,  1.10590376e-02,
                    -1.09048635e-02, -2.85400383e-02, -2.08799885e-02,  2.56681667e-02,
                    4.33091263e-02, -4.11216871e-03, -5.22413075e-02, -1.57761296e-02,
                    3.85064574e-02,  2.44861946e-02, -5.94644029e-03, -2.37867277e-02,
                    -2.08974540e-02,  6.12977218e-03,  3.41808685e-02,  2.11225797e-02,
                    -3.48074853e-02, -3.68432258e-02,  1.84538236e-02,  3.62614155e-02,
                    5.71392888e-03, -2.22282929e-02, -2.07595900e-02, -2.14059628e-03])


    Z = np.array([ 5.09056086e+00   -0.j,         -1.05632512e+00   +9.73729935j,
                    -9.07353742e-03   +5.72683175j,  7.64836214e+00  +15.08801167j,
                    -1.62335814e-03  +22.93827085j,  1.82751355e+00  +18.42963471j,
                    9.25560346e+00  +29.09247745j,  2.66648444e-01  +36.55947972j,
                    3.22763963e+00  +31.43165332j,  1.09683154e+01  +44.2430493j,
                    3.41287314e-02  +51.72758767j,  4.58707885e+00  +45.60478219j,
                    1.32002336e+01  +61.7945623j,  -7.61794544e-01  +69.7399397j,
                    6.18595526e+00  +62.00375391j,  1.65234008e+01  +83.7536092j,
                    -2.44740467e+00  +92.90549323j,  8.46397679e+00  +82.49966054j,
                    2.21742913e+01 +114.28550705j, -5.89026944e+00 +126.41635571j,
                    1.25282544e+01 +111.40965162j,  3.38317455e+01 +164.5922175j,
                    -1.36187899e+01 +186.02687058j,  2.28373276e+01 +163.09568693j,
                    6.98089809e+01 +281.33099433j, -3.41105039e+01 +359.14769618j,
                    9.25778126e+01 +348.74080231j,  8.21859482e+02+1199.51763105j,
                    1.12890247e+03-1003.57513049j, -1.01222522e+01 -186.89377673j,
                    -4.70763533e+01 -262.98269192j,  1.12069654e+02  -91.51484419j,
                    -1.54509911e+01  +54.3419452j,  -2.88583544e+01  -36.42309578j,
                    8.69028690e+01  +55.11970243j, -1.08762669e+01 +164.38367813j,
                    -1.17610319e+01  +86.57176356j,  1.18554777e+02 +213.48758446j,
                    1.20604104e+01 +395.17856585j,  9.57261799e+01 +399.31423201j,
                    1.16331747e+03 +970.86172265j,  1.04840087e+03-1020.33702175j,
                    6.54927534e+01 -429.13554842j,  2.24740868e+01 -363.85182898j,
                    6.78939098e+01 -188.45165068j, -1.98844069e+01 -125.62682656j,
                    2.05399701e+01 -141.29161717j,  3.79281797e+01  -41.10293417j,
                    -2.73052850e+01  -38.00310362j,  3.61728907e+01  -59.80486856j,])

    #Ez = np.array([])

    # fmt: on
    def test_voxelize_rectilinear(self, use_gpu):
        """
        Tests 'voxelize_rectilinear' and subpixel smoothing using the
        exact cavity and shell gridLogs configuration.
        """

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"  # logs["stl_solids"]["cavity"]
        solid_2 = "tests/stl/007_lossymetal_shell.stl"  # logs["stl_solids"]["shell"]

        stl_solids = {"cavity": solid_1, "shell": solid_2}

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

        # Number of mesh cells
        Nx = 60  # logs["Nx"]
        Ny = 60  # logs["Ny"]
        Nz = 140  # logs["Nz"]

        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,  # Global domain zmin
            zmax,  # Global domain zmax
            Nx,
            Ny,
            Nz,  # Global domain Nz
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_method="voxelize_rectilinear",
            subpixel_smoothing=False,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )

        # number of cells in the mask
        n_inside = grid.grid.threshold(scalars="shell", value=0.5).n_cells
        n_inside_expected = 61258
        assert n_inside == n_inside_expected, (
            f"Number of cells masked inside the shell is {n_inside}, expected {n_inside_expected}"
        )

        # volume
        vol = n_inside * np.min(grid.dx) * np.min(grid.dy) * np.min(grid.dz)
        vol_expected = 0.02629232100267493
        assert np.allclose(vol, vol_expected, rtol=1e-5), (
            f"Volume of the shell mask is {vol}, expected {vol_expected}"
        )

    def test_subpixel_smoothing(self):
        """
        Tests 'voxelize_rectilinear' and subpixel smoothing using the
        exact cavity and shell gridLogs configuration.
        """

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"  # logs["stl_solids"]["cavity"]
        solid_2 = "tests/stl/007_lossymetal_shell.stl"  # logs["stl_solids"]["shell"]

        stl_solids = {"cavity": solid_1, "shell": solid_2}

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

        # Number of mesh cells
        Nx = 60  # logs["Nx"]
        Ny = 60  # logs["Ny"]
        Nz = 140  # logs["Nz"]

        global grid
        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,  # Global domain zmin
            zmax,  # Global domain zmax
            Nx,
            Ny,
            Nz,  # Global domain Nz
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_method="voxelize_rectilinear",
            subpixel_smoothing=True,
            subpixel_smoothing_factor=4,
            subpixel_smoothing_bool=True,
            subpixel_smoothing_threshold=0.3,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )

        # number of cells in the mask
        n_inside = grid.grid.threshold(scalars="shell", value=0.5).n_cells
        n_inside_expected = 89256
        assert n_inside == n_inside_expected, (
            f"Number of cells masked inside the shell is {n_inside}, expected {n_inside_expected}"
        )

        # volume
        vol = n_inside * np.min(grid.dx) * np.min(grid.dy) * np.min(grid.dz)
        vol_expected = 0.038309239665264186
        assert np.allclose(vol, vol_expected, rtol=1e-5), (
            f"Volume of the shell mask is {vol}, expected {vol_expected}"
        )

    def test_long_wake_potential_and_impedance(self, use_gpu):
        global grid
        # ------------ Beam source ----------------
        # Beam parameters
        sigmaz = 10e-2  # [m] -> 2 GHz
        q = 1e-9  # [C]
        beta = 1.0  # beam beta
        xs = 0.0  # x source position [m]
        ys = 0.0  # y source position [m]
        xt = 0.0  # x test position [m]
        yt = 0.0  # y test position [m]
        # [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s]

        # ----------- Wake Solver  setup  ----------
        # Wakefield post-processor
        wakelength = 10.0  # [m] -> Partially decayed
        skip_cells = 20  # no. cells to skip at zlo/zhi for wake integration
        results_folder = "tests/013_results/"

        wake = WakeSolver(
            q=q,
            sigmaz=sigmaz,
            beta=beta,
            xsource=xs,
            ysource=ys,
            xtest=xt,
            ytest=yt,
            skip_cells=skip_cells,
            results_folder=results_folder,
            Ez_file=results_folder + "Ez.h5",
        )

        # ----------- Solver & Simulation ----------
        # boundary conditions
        bc_low = ["pec", "pec", "pec"]
        bc_high = ["pec", "pec", "pec"]

        # Solver setup
        solver = SolverFIT3D(
            grid,
            wake,
            bc_low=bc_low,
            bc_high=bc_high,
            use_stl=True,
            bg="pec",  # Background material
            dtype=self.dtype,
            use_gpu=use_gpu,
        )

        # Run simulation
        solver.wakesolve(wakelength=wakelength)

        # print(wake.WP[::50])
        np.cumsum(np.abs(wake.WP))[-1]
        assert len(wake.WP) == 5195, "Wake potential mesh samples length mismatch"
        assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
            "Wake potential mesh samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
            179.95393780891274, 0.1
        ), "Wake potential cumsum mesh failed"

        assert len(wake.Z) == 998, "Impedance samples length mismatch"
        assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **self.tol), (
            "Abs Impedance samples mesh failed"
        )
        assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **self.tol), (
            "Real Impedance samples mesh failed"
        )
        assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z), **self.tol), (
            "Imag Impedance samples mesh failed"
        )
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(
            249395.46953432143, 0.1
        ), "Abs Impedance cumsum mesh failed"
