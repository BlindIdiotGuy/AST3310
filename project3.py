import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import sys
import scipy.constants as sc
from tqdm import tqdm
import astropy.constants as c
# visualiser
import FVis3 as FVis
from scipy import signal

class convection2D:

    def __init__(self, nabla, perturbation=False):
        """
        define variables
        """
        self.T0 = 5778          # K
        self.P0 = 1.8e4         # Pa
        self.nabla = nabla      # Temperature gradient
        self.perturbation = perturbation
        self.mu = 0.61          # Mean molecular mass
        self.kb = c.k_B.value   # Boltzmann constant
        self.m_u = c.u.value  # Atomic mass unit
        self.R_sun = 6.96e8     # m
        self.M_sun = 1.989e30   # kg
        self.G = c.G.value      # Gravitational constant
        self.g = np.abs(self.G * self.M_sun / (self.R_sun**2))
        self.gamma = 5/3        # Adiabatic gas constant
        self.x_max = 12e6           # m
        self.y_max = 4e6            # m
        self.Nx = 300
        self.Ny = 100
        self.x = np.linspace(0, self.x_max, self.Nx)
        self.y = np.linspace(0, self.y_max, self.Ny)
        self.dx = self.x_max / self.Nx
        self.dy = self.y_max / self.Ny
        self.dt = 0.1

    def T(self, y):
        """
        calculates temperature
        """
        a = self.nabla * self.mu * self.m_u * self.g * (y - self.y_max)
        b = self.kb * self.T0
        T = self.T0 * (1 - a / b)
        return T
    
    def P(self, y, T):
        """
        calculates pressure
        """
        P = ((self.P0**self.nabla / self.T0) * T)**(1/self.nabla)
        return P

    def create_gaussian_perturbation(self, center_x, center_y, sigma, A):
        X, Y = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
        kernel = A * np.exp(-0.5 * ((X - center_x)**2 + (Y - center_y)**2) / sigma**2)
        return kernel
        

    def initialise(self):
        """
        initialise temperature, pressure, density and internal energy
        """
        X, Y = np.meshgrid(self.x, self.y)
        if self.perturbation == False:
            self.T_array = self.T(Y)
        elif self.perturbation == True:
            self.T_array = self.T(Y) + self.create_gaussian_perturbation(self.Nx/2, self.Ny/4, 10, 40000)
        # self.P_array = self.P(Y, self.T_array)
        self.P_array = self.P0 * (self.T_array / self.T0)**(1 / self.nabla)
        self.e_array = self.P_array / (self.gamma - 1)
        self.rho_array = (self.gamma - 1) * (self.mu * self.m_u) / (self.kb * self.T_array) * self.e_array
        self.u_array = np.zeros((self.Ny, self.Nx))
        self.w_array = np.zeros((self.Ny, self.Nx))

        self.dedt_array = np.zeros((self.Ny, self.Nx))
        self.drhodt_array = np.zeros((self.Ny, self.Nx))
        self.drhoudt_array = -self.central_x(self.P_array)
        self.drhowdt_array = -self.central_y(self.P_array) + self.rho_array * self.g
        # print(f'dedt = {self.dedt_array}, drhodt = {self.drhodt_array}, drhoudt = {self.drhoudt_array}, drhowdt = {self.drhowdt_array}')
        # print(f'P = {self.P_array}, dx = {self.dx}, g = {self.g}')
        
    def rel_phi(self, var, dvar_dt):
        """
        calculates the relative change of a variable
        """
        relphi = dvar_dt / var
        return relphi
    
    def dedt(self, u, w, e):
        """
        calculates the time derivative of energy
        """
        P = (self.gamma - 1) * e
        de_dt = -e * (self.central_x(u) + self.central_y(w)) - u * self.upwind_x(e, u) - w * self.upwind_y(e, w) - P * (self.central_x(u) + self.central_y(w))
        return de_dt

    def drhodt(self, u, w, rho):
        """
        calculates the time derivative of density
        """
        drho_dt = -rho * (self.central_x(u) + self.central_y(w)) - u * self.upwind_x(rho, u) - w * self.upwind_y(rho, w)
        return drho_dt
    
    def drhoudt(self, u, w, rho, P):
        """
        calculates the time derivative of momentum in x-direction
        """
        drhou_dt = -rho * u * (self.upwind_x(u, u) + self.central_y(w)) - u * self.upwind_x(rho * u, u) - w * self.upwind_y(rho * u, w) - self.central_x(P)
        return drhou_dt
    
    def drhowdt(self, u, w, rho, P):
        """
        calculates time derivative of momentum in y-direction
        """
        drhow_dt = -rho * w * (self.central_x(u) + self.upwind_y(w, w)) - w * self.upwind_y(rho * w, w) - u * self.upwind_x(rho * w, u) - self.central_y(P) - rho * self.g
        return drhow_dt

    def timestep(self):
        """
        calculate timestep
        """
        p = 0.1

        rel_x = np.abs(self.u_array / self.dx)
        rel_y = np.abs(self.w_array / self.dy)
        rel_e = np.abs(self.rel_phi(self.e_array, self.dedt_array))
        rel_rho = np.abs(self.rel_phi(self.rho_array, self.drhodt_array))
        rel_rhou = np.abs(self.rel_phi(self.rho_array * self.u_array, self.drhoudt_array))
        rel_rhow = np.abs(self.rel_phi(self.rho_array * self.w_array, self.drhowdt_array))

        self.delta = np.nanmax([np.nanmax(rel_e), np.nanmax(rel_rho), np.nanmax(rel_rhou), np.nanmax(rel_rhow), np.nanmax(rel_x), np.nanmax(rel_y)])
        # if self.delta > 100:
        #     self.delta = 100
        # self.dt = p / self.delta

        # # if np.isinf(self.delta):
        # #     print(f'delta is inf')

        # if self.dt < 1e-3 or np.isinf(self.dt) or np.isnan(self.dt):
        #     self.dt = 1e-3
        if 0.01 <= self.delta <= 100:
            self.dt = p / self.delta
            return self.dt
        else:
            return 0.001

    def boundary_conditions(self):
        """
        boundary conditions for energy, density and velocity
        """
        self.alpha_top = -self.g * self.mu * self.m_u / (self.kb * self.T_array[-1, :])
        self.alpha_bottom = -self.g * self.mu * self.m_u / (self.kb * self.T_array[0, :])
        e_top = (4 * self.e_array[-2, :] - self.e_array[-3, :]) / (3 + 2*self.dy * self.alpha_top) 
        e_bottom = (4 * self.e_array[1, :] - self.e_array[2, :]) / (3 - 2*self.dy * self.alpha_bottom)
        self.e_array[-1, :] = e_top
        self.e_array[0, :] = e_bottom
        self.rho_array[-1, :] = (self.gamma - 1) * (self.mu * self.m_u) / (self.kb * self.T_array[-1, :]) * self.e_array[-1, :]
        self.rho_array[0, :] = (self.gamma - 1) * (self.mu * self.m_u) / (self.kb * self.T_array[0, :]) * self.e_array[0, :]
        self.u_array[-1, :] = 0
        self.u_array[0, :] = 0
        self.w_array[-1, :] = 0
        self.w_array[0, :] = 0

    def central_x(self, var):
        """
        central difference scheme in x-direction
        """
        var_right = np.roll(var, -1, axis=1)
        var_left = np.roll(var, 1, axis=1)
        return (var_right - var_left) / (2 * self.dx)

    def central_y(self, var):
        """
        central difference scheme in y-direction
        """
        var_up = np.roll(var, -1, axis=0)
        var_down = np.roll(var, 1, axis=0)
        return (var_up - var_down) / (2 * self.dy)

    def upwind_x(self, var, v):
        """
        upwind difference scheme in x-direction
        """
        return np.where(v>=0, (var - np.roll(var, 1, axis=1)) / self.dx, (np.roll(var, -1, axis=1) - var) / self.dx)

    def upwind_y(self, var, v):
        """
        upwind difference scheme in y-direction
        """
        return np.where(v>=0, (var - np.roll(var, 1, axis=0)) / self.dy, (np.roll(var, -1, axis=0) - var) / self.dy)

    def hydro_solver(self):
        """
        hydrodynamic equations solver
        """
        self.timestep()
        self.dedt_array = self.dedt(self.u_array, self.w_array, self.e_array)
        self.drhodt_array = self.drhodt(self.u_array, self.w_array, self.rho_array)
        self.drhoudt_array = self.drhoudt(self.u_array, self.w_array, self.rho_array, self.P_array)
        self.drhowdt_array = self.drhowdt(self.u_array, self.w_array, self.rho_array, self.P_array)
    
        self.e_array += self.dedt_array * self.dt
        self.rho_array += self.drhodt_array * self.dt
        self.u_array += self.drhoudt_array * self.dt
        self.w_array += self.drhowdt_array * self.dt
        self.boundary_conditions()
        self.P_array = (self.gamma - 1) * self.e_array
        self.T_array = (self.mu * self.m_u) / (self.rho_array * self.kb) * self.P_array

        dt = self.dt
        return dt
    def print_dt(self):
        print(f'dt = {self.dt}, delta = {self.delta}')
    

clas = convection2D(2/5, True)
clas.initialise()
clas.boundary_conditions()
plt.imshow(clas.T_array, origin='lower', extent=(0, clas.x_max, 0, clas.y_max), aspect='auto', cmap='inferno')
plt.colorbar(label='Temperature (K)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Temperature distribution')
plt.show()

vis = FVis.FluidVisualiser()
solver = convection2D(2/5, perturbation=True)
solver.initialise()
solver.hydro_solver()
solver.print_dt()
vis.save_data(60, solver.hydro_solver, rho=solver.rho_array, e=solver.e_array, u=solver.u_array, w=solver.w_array, T=solver.T_array, P=solver.P_array)
vis.animate_2D('T', anim_fps=27, anim_time=60, folder = 'FVis_output_true')

def sanity_check():
    """
    sanity check for the hydro solver
    """
    vis = FVis.FluidVisualiser()
    solver = convection2D(2/5)
    solver.initialise()
    vis.save_data(60, solver.hydro_solver, rho=solver.rho_array, e=solver.e_array, u=solver.u_array, w=solver.w_array, T=solver.T_array, P=solver.P_array)
    vis.animate_2D('rho')
    vis.animate_2D('e')
    vis.animate_2D('u')
    vis.animate_2D('w')
    vis.animate_2D('P')
    vis.animate_2D('T')
    vis.delete_current_data()