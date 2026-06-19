# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np
from scipy.constants import mu_0 as mu_0
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import c as c_light
from scipy.sparse import diags

from .field import Field


class BCsMixin:
    def _apply_bc_to_C(self):
        """
        Apply boundary conditions by modifying curl and metric matrices.

        Adjusts rows/columns of the curl operator ``C`` and the metric-diagonal
        matrices (``tDs``, ``itDa``) according to the low/high boundary
        condition lists ``bc_low`` and ``bc_high``. Handles periodic, PEC/PMC,
        ABC and PML options and also configures MPI-internal faces when the
        grid is subdivided.
        """
        xlo, ylo, zlo = 1.0, 1.0, 1.0
        xhi, yhi, zhi = 1.0, 1.0, 1.0

        # Check BCs for internal MPI subdomains
        if self.use_mpi and self.grid.use_mpi:
            if self.rank > 0:
                self.bc_low = ["pec", "pec", "mpi"]

            if self.rank < self.size - 1:
                self.bc_high = ["pec", "pec", "mpi"]

        # Perodic: out == in
        if any(True for x in self.bc_low if x.lower() == "periodic"):
            if (
                self.bc_low[0].lower() == "periodic"
                and self.bc_high[0].lower() == "periodic"
            ):
                self.tL[-1, :, :, "x"] = self.L[0, :, :, "x"]
                self.itA[-1, :, :, "y"] = self.iA[0, :, :, "y"]
                self.itA[-1, :, :, "z"] = self.iA[0, :, :, "z"]

            if (
                self.bc_low[1].lower() == "periodic"
                and self.bc_high[1].lower() == "periodic"
            ):
                self.tL[:, -1, :, "y"] = self.L[:, 0, :, "y"]
                self.itA[:, -1, :, "x"] = self.iA[:, 0, :, "x"]
                self.itA[:, -1, :, "z"] = self.iA[:, 0, :, "z"]

            if (
                self.bc_low[2].lower() == "periodic"
                and self.bc_high[2].lower() == "periodic"
            ):
                self.tL[:, :, -1, "z"] = self.L[:, :, 0, "z"]
                self.itA[:, :, -1, "x"] = self.iA[:, :, 0, "x"]
                self.itA[:, :, -1, "y"] = self.iA[:, :, 0, "y"]

            self.tDs = diags(
                self.tL.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.itDa = diags(
                self.itA.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )

        # Dirichlet PEC: tangential E field = 0 at boundary
        if any(
            True
            for x in self.bc_low
            if x.lower() in ("electric", "pec", "pml")
        ) or any(
            True
            for x in self.bc_high
            if x.lower() in ("electric", "pec", "pml")
        ):
            if self.bc_low[0].lower() in ("electric", "pec", "pml"):
                xlo = 0
            if self.bc_low[1].lower() in ("electric", "pec", "pml"):
                ylo = 0
            if self.bc_low[2].lower() in ("electric", "pec", "pml"):
                zlo = 0
            if self.bc_high[0].lower() in ("electric", "pec", "pml"):
                xhi = 0
            if self.bc_high[1].lower() in ("electric", "pec", "pml"):
                yhi = 0
            if self.bc_high[2].lower() in ("electric", "pec", "pml"):
                zhi = 0

            # Assemble matrix
            self.BC = Field(
                self.Nx, self.Ny, self.Nz, dtype=np.int8, use_ones=True
            )

            for d in ["x", "y", "z"]:  # tangential to zero
                if d != "x":
                    self.BC[0, :, :, d] = xlo
                    self.BC[-1, :, :, d] = xhi
                if d != "y":
                    self.BC[:, 0, :, d] = ylo
                    self.BC[:, -1, :, d] = yhi
                if d != "z":
                    self.BC[:, :, 0, d] = zlo
                    self.BC[:, :, -1, d] = zhi

            self.Dbc = diags(
                self.BC.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=np.int8,
            )
            self.Dbc_x = diags(
                self.BC.field_x,
                shape=(self.N, self.N),
                dtype=np.int8,
            )
            self.Dbc_y = diags(
                self.BC.field_y,
                shape=(self.N, self.N),
                dtype=np.int8,
            )
            self.Dbc_z = diags(
                self.BC.field_z,
                shape=(self.N, self.N),
                dtype=np.int8,
            )

            # Update C (columns)
            self.C = self.C * self.Dbc

        # Dirichlet PMC: tangential H field = 0 at boundary
        if any(
            True for x in self.bc_low if x.lower() in ("magnetic", "pmc")
        ) or any(
            True for x in self.bc_high if x.lower() in ("magnetic", "pmc")
        ):
            if self.bc_low[0].lower() == "magnetic" or self.bc_low[0] == "pmc":
                xlo = 0
            if self.bc_low[1].lower() == "magnetic" or self.bc_low[1] == "pmc":
                ylo = 0
            if self.bc_low[2].lower() == "magnetic" or self.bc_low[2] == "pmc":
                zlo = 0
            if (
                self.bc_high[0].lower() == "magnetic"
                or self.bc_high[0] == "pmc"
            ):
                xhi = 0
            if (
                self.bc_high[1].lower() == "magnetic"
                or self.bc_high[1] == "pmc"
            ):
                yhi = 0
            if (
                self.bc_high[2].lower() == "magnetic"
                or self.bc_high[2] == "pmc"
            ):
                zhi = 0

            # Assemble matrix
            self.BC = Field(
                self.Nx, self.Ny, self.Nz, dtype=np.int8, use_ones=True
            )

            for d in ["x", "y", "z"]:  # tangential to zero
                if d != "x":
                    self.BC[0, :, :, d] = xlo
                    self.BC[-1, :, :, d] = xhi
                if d != "y":
                    self.BC[:, 0, :, d] = ylo
                    self.BC[:, -1, :, d] = yhi
                if d != "z":
                    self.BC[:, :, 0, d] = zlo
                    self.BC[:, :, -1, d] = zhi

            self.Dbc = diags(
                self.BC.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=np.int8,
            )
            self.Dbc_x = diags(
                self.BC.field_x,
                shape=(self.N, self.N),
                dtype=np.int8,
            )
            self.Dbc_y = diags(
                self.BC.field_y,
                shape=(self.N, self.N),
                dtype=np.int8,
            )
            self.Dbc_z = diags(
                self.BC.field_z,
                shape=(self.N, self.N),
                dtype=np.int8,
            )

            # Update C (rows)
            self.C = self.Dbc * self.C

        # Absorbing boundary conditions ABC
        if any(True for x in self.bc_low if x.lower() == "abc") or any(
            True for x in self.bc_high if x.lower() == "abc"
        ):
            if self.bc_high[0].lower() == "abc":
                self.tL[-1, :, :, "x"] = self.L[0, :, :, "x"]
                self.itA[-1, :, :, "y"] = self.iA[0, :, :, "y"]
                self.itA[-1, :, :, "z"] = self.iA[0, :, :, "z"]

            if self.bc_high[1].lower() == "abc":
                self.tL[:, -1, :, "y"] = self.L[:, 0, :, "y"]
                self.itA[:, -1, :, "x"] = self.iA[:, 0, :, "x"]
                self.itA[:, -1, :, "z"] = self.iA[:, 0, :, "z"]

            if self.bc_high[2].lower() == "abc":
                self.tL[:, :, -1, "z"] = self.L[:, :, 0, "z"]
                self.itA[:, :, -1, "x"] = self.iA[:, :, 0, "x"]
                self.itA[:, :, -1, "y"] = self.iA[:, :, 0, "y"]

            self.tDs = diags(
                self.tL.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.itDa = diags(
                self.itA.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.activate_abc = True

        # Perfect Matching Layers (PML)
        if any(True for x in self.bc_low if x.lower() == "pml") or any(
            True for x in self.bc_high if x.lower() == "pml"
        ):
            self.activate_pml = True
            #self.use_conductivity = True

    def _initialize_PML(self):
        """
        Compute and apply PML sigma profiles to the solver conductivity tensor.

        Uses configured PML settings (number of layers, profile function and
        scaling) to set per-component conductivity in the PML regions. This is
        used to absorb outgoing waves and reduce reflections at domain edges.
        """

        # Initialize PML parameters
        if self.verbose>1:
            print("Initializing PML parameters...")
        R0 = 1.0e-8        # Reflection coefficient at the interface between the PML and the main domain, controls how well the PML absorbs waves (lower is better but may require stronger conductivity)
        eta_0 = 376.730313412    
        sx, sy, sz = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        ax, ay, az = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        tsx, tsy, tsz = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        tax, tay, taz = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        self.kappa = (
            Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
        )
        self.alpha = (
            Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        )
        self.sigma_pml = (
            Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        )
        self.tkappa = (
            Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
        )
        self.talpha = (
            Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        )
        self.tsigma_pml = (
            Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        )
        sigma_pml_x, sigma_pml_y, sigma_pml_z = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        kappa_x, kappa_y, kappa_z = np.ones(self.Nx), np.ones(self.Ny), np.ones(self.Nz)
        alpha_x, alpha_y, alpha_z = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        tsigma_pml_x, tsigma_pml_y, tsigma_pml_z = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        tkappa_x, tkappa_y, tkappa_z = np.ones(self.Nx), np.ones(self.Ny), np.ones(self.Nz)
        talpha_x, talpha_y, talpha_z = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)

        # Fill
        if self.bc_low[0].lower() == "pml":
            interface = self.x[self.n_pml]
            L = interface - self.x[0]
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(self.n_pml):
                dist = interface - self.x[i]   # distance into PML
                tdist = interface - (self.x[i] + self.dx[i]/2)   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sx[i] = (dist / L)**self.pml_exp
                ax[i] = (dist / L)
                tax[i] = (tdist / L)
                tsx[i] = (tdist / L)**self.pml_exp
                sigma_pml_x[i] = sigma_max * sx[i]
                kappa_x[i] = 1 + (self.kappa_max - 1) * sx[i]
                alpha_x[i] = self.alpha_max * (1 - ax[i])
                tsigma_pml_x[i] = sigma_max * tsx[i]
                tkappa_x[i] = 1 + (self.kappa_max - 1) * tsx[i]
                talpha_x[i] = self.alpha_max * (1 - tax[i])

        if self.bc_low[1].lower() == "pml":
            interface = self.y[self.n_pml]
            L = interface - self.y[0]
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(self.n_pml):
                dist = interface - self.y[i]   # distance into PML
                tdist = interface - (self.y[i] + self.dy[i]/2)   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sy[i] = (dist / L)**self.pml_exp
                tsy[i] = (tdist / L)**self.pml_exp
                ay[i] = (dist / L)
                tay[i] = (tdist / L)
                sigma_pml_y[i] = sigma_max * sy[i]
                kappa_y[i] = 1 + (self.kappa_max - 1) * sy[i]
                alpha_y[i] = self.alpha_max * (1 - ay[i])
                tsigma_pml_y[i] = sigma_max * tsy[i]
                tkappa_y[i] = 1 + (self.kappa_max - 1) * tsy[i]
                talpha_y[i] = self.alpha_max * (1 - tay[i])     

        if self.bc_low[2].lower() == "pml":
            interface = self.z[self.n_pml]
            L = interface - self.z[0]
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(self.n_pml):
                dist = interface - self.z[i]   # distance into PML
                tdist = interface - (self.z[i] + self.dz[i]/2)   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sz[i] = (dist / L)**self.pml_exp
                tsz[i] = (tdist / L)**self.pml_exp
                az[i] = (dist / L)
                taz[i] = (tdist / L)
                sigma_pml_z[i] = sigma_max * sz[i]
                kappa_z[i] = 1 + (self.kappa_max - 1) * sz[i]
                alpha_z[i] = self.alpha_max * (1 - az[i])
                tsigma_pml_z[i] = sigma_max * tsz[i]
                tkappa_z[i] = 1 + (self.kappa_max - 1) * tsz[i]
                talpha_z[i] = self.alpha_max * (1 - taz[i])

        if self.bc_high[0].lower() == "pml":
            interface = self.x[-1-self.n_pml]
            L = self.x[-1] - interface
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(-self.n_pml-1, 0):
                dist = self.x[i] - interface   # distance into PML
                tdist = (self.x[i] - self.dx[i]/2) - interface   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sx[i] = (dist / L)**self.pml_exp
                tsx[i] = (tdist / L)**self.pml_exp
                ax[i] = (dist / L)
                tax[i] = (tdist / L)
                sigma_pml_x[i] = sigma_max * sx[i]
                kappa_x[i] = 1 + (self.kappa_max - 1) * sx[i]
                alpha_x[i] = self.alpha_max * (1 - ax[i])
                tsigma_pml_x[i] = sigma_max * tsx[i]
                tkappa_x[i] = 1 + (self.kappa_max - 1) * tsx[i]
                talpha_x[i] = self.alpha_max * (1 - tax[i]) 

        if self.bc_high[1].lower() == "pml":
            interface = self.y[-1-self.n_pml]
            L = self.y[-1] - interface
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(-self.n_pml-1, 0):
                dist = self.y[i] - interface   # distance into PML
                tdist = (self.y[i] + self.dy[i]/2) - interface   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sy[i] = (dist / L)**self.pml_exp
                tsy[i] = (tdist / L)**self.pml_exp
                ay[i] = (dist / L)
                tay[i] = (tdist / L)
                sigma_pml_y[i] = sigma_max * sy[i]
                kappa_y[i] = 1 + (self.kappa_max - 1) * sy[i]
                alpha_y[i] = self.alpha_max * (1 - ay[i])
                tsigma_pml_y[i] = sigma_max * tsy[i]
                tkappa_y[i] = 1 + (self.kappa_max - 1) * tsy[i]
                talpha_y[i] = self.alpha_max * (1 - tay[i]) 

        if self.bc_high[2].lower() == "pml":
            interface = self.z[-1-self.n_pml]
            L = self.z[-1] - interface
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)

            for i in range(-self.n_pml-1, 0):
                dist = self.z[i] - interface   # distance into PML
                sz[i] = (dist / L)**self.pml_exp
                az[i] = (dist / L)
                sigma_pml_z[i] = sigma_max * sz[i]
                kappa_z[i] = 1 + (self.kappa_max - 1) * sz[i]
                alpha_z[i] = self.alpha_max * (1 - az[i])

                tdist = self.z[i] + self.dz[i]/2 - interface   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                tsz[i] = (tdist / L)**self.pml_exp
                taz[i] = (tdist / L)
                tsigma_pml_z[i] = sigma_max * tsz[i]
                tkappa_z[i] = 1 + (self.kappa_max - 1) * tsz[i]
                talpha_z[i] = self.alpha_max * (1 - taz[i]) 

        self.sigma_pml[:, :, :, 'x'] = sigma_pml_x[:, np.newaxis, np.newaxis]
        self.sigma_pml[:, :, :, 'y'] = sigma_pml_y[np.newaxis, :, np.newaxis]
        self.sigma_pml[:, :, :, 'z'] = sigma_pml_z[np.newaxis, np.newaxis, :]
        self.tsigma_pml[:, :, :, 'x'] = tsigma_pml_x[:, np.newaxis, np.newaxis]
        self.tsigma_pml[:, :, :, 'y'] = tsigma_pml_y[np.newaxis, :, np.newaxis]
        self.tsigma_pml[:, :, :, 'z'] = tsigma_pml_z[np.newaxis, np.newaxis, :]
        self.kappa[:, :, :, 'x'] = kappa_x[:, np.newaxis, np.newaxis]
        self.kappa[:, :, :, 'y'] = kappa_y[np.newaxis, :, np.newaxis]
        self.kappa[:, :, :, 'z'] = kappa_z[np.newaxis, np.newaxis, :]
        self.tkappa[:, :, :, 'x'] = tkappa_x[:, np.newaxis, np.newaxis]
        self.tkappa[:, :, :, 'y'] = tkappa_y[np.newaxis, :, np.newaxis]
        self.tkappa[:, :, :, 'z'] = tkappa_z[np.newaxis, np.newaxis, :]
        self.alpha[:, :, :, 'x'] = alpha_x[:, np.newaxis, np.newaxis]
        self.alpha[:, :, :, 'y'] = alpha_y[np.newaxis, :, np.newaxis]
        self.alpha[:, :, :, 'z'] = alpha_z[np.newaxis, np.newaxis, :]
        self.talpha[:, :, :, 'x'] = talpha_x[:, np.newaxis, np.newaxis]
        self.talpha[:, :, :, 'y'] = talpha_y[np.newaxis, :, np.newaxis]
        self.talpha[:, :, :, 'z'] = talpha_z[np.newaxis, np.newaxis, :]

    def get_abc(self):
        """
        Save boundary field snapshots needed by the Absorbing Boundary
        Condition (ABC) update.

        Extracts the necessary boundary layers for electric and magnetic
        fields for those faces configured with ABC and returns two
        dictionaries holding the saved arrays. Those dictionaries are later
        consumed by ``update_abc`` to restore boundary values.
        """
        E_abc, H_abc = {}, {}

        if self.bc_low[0].lower() == "abc":
            E_abc[0] = {}
            H_abc[0] = {}
            for d in ["x", "y", "z"]:
                E_abc[0][d + "lo"] = self.E[1, :, :, d]
                H_abc[0][d + "lo"] = self.H[1, :, :, d]

        if self.bc_low[1].lower() == "abc":
            E_abc[1] = {}
            H_abc[1] = {}
            for d in ["x", "y", "z"]:
                E_abc[1][d + "lo"] = self.E[:, 1, :, d]
                H_abc[1][d + "lo"] = self.H[:, 1, :, d]

        if self.bc_low[2].lower() == "abc":
            E_abc[2] = {}
            H_abc[2] = {}
            for d in ["x", "y", "z"]:
                E_abc[2][d + "lo"] = self.E[:, :, 1, d]
                H_abc[2][d + "lo"] = self.H[:, :, 1, d]

        if self.bc_high[0].lower() == "abc":
            E_abc[0] = {}
            H_abc[0] = {}
            for d in ["x", "y", "z"]:
                E_abc[0][d + "hi"] = self.E[-1, :, :, d]
                H_abc[0][d + "hi"] = self.H[-1, :, :, d]

        if self.bc_high[1].lower() == "abc":
            E_abc[1] = {}
            H_abc[1] = {}
            for d in ["x", "y", "z"]:
                E_abc[1][d + "hi"] = self.E[:, -1, :, d]
                H_abc[1][d + "hi"] = self.H[:, -1, :, d]

        if self.bc_high[2].lower() == "abc":
            E_abc[2] = {}
            H_abc[2] = {}
            for d in ["x", "y", "z"]:
                E_abc[2][d + "hi"] = self.E[:, :, -1, d]
                H_abc[2][d + "hi"] = self.H[:, :, -1, d]

        return E_abc, H_abc

    def update_abc(self, E_abc, H_abc):
        """
        Apply the Absorbing Boundary Condition (ABC) using previously saved
        snapshots.

        Parameters
        ----------
        E_abc, H_abc : dict
            Dictionaries produced by ``get_abc`` that contain boundary-layer
            field arrays. Each dictionary maps face indices to arrays used to
            overwrite the exterior cell values after a timestep.
        """

        if self.bc_low[0].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[0, :, :, d] = E_abc[0][d + "lo"]
                self.H[0, :, :, d] = H_abc[0][d + "lo"]

        if self.bc_low[1].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[:, 0, :, d] = E_abc[1][d + "lo"]
                self.H[:, 0, :, d] = H_abc[1][d + "lo"]

        if self.bc_low[2].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[:, :, 0, d] = E_abc[2][d + "lo"]
                self.H[:, :, 0, d] = H_abc[2][d + "lo"]

        if self.bc_high[0].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[-1, :, :, d] = E_abc[0][d + "hi"]
                self.H[-1, :, :, d] = H_abc[0][d + "hi"]

        if self.bc_high[1].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[:, -1, :, d] = E_abc[1][d + "hi"]
                self.H[:, -1, :, d] = H_abc[1][d + "hi"]

        if self.bc_high[2].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[:, :, -1, d] = E_abc[2][d + "hi"]
                self.H[:, :, -1, d] = H_abc[2][d + "hi"]
