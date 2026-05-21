import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
# visualiser
import FVis3 as FVis

#-----------------------------------------------------------------------------------------------------------#\
check_sanity = False         # Bool to decide if the sanity checks are on or off, True for on, False for off |
#-----------------------------------------------------------------------------------------------------------#/

class convection2D:

    def __init__(self, nabla:float = 2/5, perturbations:int = 0):
        """
        Define variables

        Parameters
        ----------
        nabla : float, optional
            Temperature gradient, default is 2/5
        perturbations : int, optional
            Number of perturbations, 0, 1, 3, and 5 are valid options, 0 is the default
        """
        self.T0 = 5778                      # K
        self.P0 = 1.8e4                     # Pa
        self.nabla = nabla                  # Temperature gradient
        self.perturbations = perturbations  # Whether to add perturbation to the initial temperature distribution
        self.mu = 0.61                      # Mean molecular mass
        self.kb = c.k_B.value               # Boltzmann constant
        self.m_u = c.u.value                # Atomic mass unit
        self.R_sun = 6.96e8                 # m
        self.M_sun = 1.989e30               # kg
        self.G = c.G.value                  # Gravitational constant
        self.g = np.abs(self.G * self.M_sun / (self.R_sun**2))  # Gravitational acceleration, m s^-2
        self.gamma = 5/3                    # Adiabatic gas constant
        self.x_max = 12e6                   # m
        self.y_max = 4e6                    # m
        self.Nx = 300                       # Number of grid points in x-direction
        self.Ny = 100                       # Number of grid points in y-direction
        self.x, self.dx = np.linspace(0, self.x_max, self.Nx, retstep=True)
        self.y, self.dy = np.linspace(0, self.y_max, self.Ny, retstep=True)
        self.dt = 0.1                       # Initial timestep, s

    def calculate_T(self, y:np.ndarray):
        """
        Calculates temperature

        Parameter
        ---------
        y : array-like
            Array of y-coordinates
        
        Returns
        -------
        T : array-like
            Temperature distribution as a function of y
        """
        a = self.nabla * self.mu * self.m_u * self.g * (y - self.y_max)
        b = self.kb * self.T0
        T = self.T0 * (1 - a / b)
        return T
    
    def calculate_P(self, y:np.ndarray):
        """
        Calculates pressure
        
        Parameter
        ---------
        y : array-like
            Array of y-coordinates
        
        Returns
        -------
        P : array-like
            Pressure distribution as a function of y
        """
        P = self.P0*(1 - self.nabla*((self.mu*self.m_u)/(self.kb))*((self.g*(y - self.y_max))/(self.T0)))**(1/self.nabla)
        return P

    def create_gaussian_perturbation(self, center_x:float, center_y:float, sigma:float, A:float):
        """
        Creates a gaussian kernel with a given center, sigma and amplitude

        Parameters
        ----------
        center_x : float
            x-coordinate of the center of the gaussian
        center_y : float
            y-coordinate of the center of the gaussian
        sigma : float
            Standard deviation of the gaussian
        A : float
            Amplitude of the gaussian

        Returns
        -------
        kernel : array-like
            Gaussian kernel with the given parameters
        """
        X, Y = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
        kernel = A * np.exp(-0.5 * ((X - center_x)**2 + (Y - center_y)**2) / sigma**2)
        return kernel
        

    def initialise(self):
        """
        Initialise temperature, pressure, density and internal energy
        """
        X, Y = np.meshgrid(self.x, self.y)
        if self.perturbations == 0:
            self.T = self.calculate_T(Y)
        elif self.perturbations == 1:
            self.T = self.calculate_T(Y) + self.create_gaussian_perturbation(self.Nx/2, self.Ny/5, 10, 5 * self.T0)
        elif self.perturbations == 3:
            self.T = self.calculate_T(Y) + self.create_gaussian_perturbation(self.Nx/2, self.Ny/5, 10, 5 * self.T0)\
                        + self.create_gaussian_perturbation(self.Nx/6, self.Ny/5, 10, 5 * self.T0) + self.create_gaussian_perturbation(5 * self.Nx/6, self.Ny/5, 10, 5 * self.T0)
        elif self.perturbations == 5:
            self.T = self.calculate_T(Y) + self.create_gaussian_perturbation(self.Nx/2, self.Ny/5, 10, 5 * self.T0)\
                        + self.create_gaussian_perturbation(self.Nx/3, self.Ny/5, 10, 5 * self.T0) + self.create_gaussian_perturbation(2 * self.Nx/3, self.Ny/5, 10, 5 * self.T0)\
                        + self.create_gaussian_perturbation(self.Nx/6, self.Ny/5, 10, 5 * self.T0) + self.create_gaussian_perturbation(5 * self.Nx/6, self.Ny/5, 10, 5 * self.T0)
        self.P = self.calculate_P(Y)
        self.e = self.P / (self.gamma - 1)
        self.rho = (self.gamma - 1) * self.e * (self.mu * self.m_u) / (self.kb * self.T)
        self.u = np.zeros((self.Ny, self.Nx))
        self.w = np.zeros((self.Ny, self.Nx))
        
    def rel_phi(self, var:np.ndarray, dvar_dt:np.ndarray):
        """
        Calculates the relative change of a variable

        Parameters
        ----------
        var : array-like
            Variable to calculate the relative change of
        dvar_dt : array-like
            Time derivative of the variable
        
        Returns
        -------
        relphi : array-like
            Relative change of the variable
        """
        relphi = dvar_dt / var
        return relphi
    
    def get_dedt(self, u:np.ndarray, w:np.ndarray, e:np.ndarray):
        """
        Calculates the time derivative of energy

        Parameters
        ----------
        u : array-like
            Velocity in x-direction
        w : array-like
            Velocity in y-direction
        e : array-like
            Internal energy
        """
        P = (self.gamma - 1) * e
        de_dt = -e * (self.central_x(u) + self.central_y(w)) - u * self.upwind_x(e, u) - w * self.upwind_y(e, w) - P * (self.central_x(u) + self.central_y(w))
        self.dedt = de_dt

    def get_drhodt(self, u:np.ndarray, w:np.ndarray, rho:np.ndarray):
        """
        Calculates the time derivative of density

        Parameters
        ----------
        u : array-like
            Velocity in x-direction
        w : array-like
            Velocity in y-direction
        rho : array-like
            Density
        """
        drho_dt = -rho * (self.central_x(u) + self.central_y(w)) - u * self.upwind_x(rho, u) - w * self.upwind_y(rho, w)
        self.drhodt =  drho_dt
    
    def get_drhoudt(self, u:np.ndarray, w:np.ndarray, rho:np.ndarray, P:np.ndarray):
        """
        Calculates the time derivative of momentum in x-direction

        Parameters
        ----------
        u : array-like
            Velocity in x-direction
        w : array-like
            Velocity in y-direction
        rho : array-like
            Density
        P : array-like
            Pressure
        """
        drhou_dt = -rho * u * (self.upwind_x(u, u) + self.central_y(w)) - u * self.upwind_x(rho * u, u) - w * self.upwind_y(rho * u, w) - self.central_x(P)
        self.drhoudt =  drhou_dt
    
    def get_drhowdt(self, u:np.ndarray, w:np.ndarray, rho:np.ndarray, P:np.ndarray):
        """
        Calculates time derivative of momentum in y-direction
        
        Parameters
        ----------
        u : array-like
            Velocity in x-direction
        w : array-like
            Velocity in y-direction
        rho : array-like
            Density
        P : array-like
            Pressure
        """
        drhow_dt = -rho * w * (self.central_x(u) + self.upwind_y(w, w)) - w * self.upwind_y(rho * w, w) - u * self.upwind_x(rho * w, u) - self.central_y(P) - rho * self.g
        self.drhowdt = drhow_dt

    def timestep(self):
        """
        Calculate timestep
        """
        p = 0.001

        max_rel_x = np.nanmax(np.abs(self.u / self.dx))
        max_rel_y = np.nanmax(np.abs(self.w / self.dy))
        max_rel_e = np.nanmax(np.abs(self.rel_phi(self.e, self.dedt)))
        max_rel_rho = np.nanmax(np.abs(self.rel_phi(self.rho, self.drhodt)))
        max_rel_rhou = np.nanmax(np.abs(self.rel_phi(self.rho * self.u, self.drhoudt)))
        max_rel_rhow = np.nanmax(np.abs(self.rel_phi(self.rho * self.w, self.drhowdt)))
        if np.isfinite(max_rel_rhou) == True:
            max_rel_rhou = max_rel_rhou
        else: 
            max_rel_rhou = 0.0001
        if np.isfinite(max_rel_rhow) == True:
            max_rel_rhow = max_rel_rhow
        else: 
            max_rel_rhow = 0.0001
        self.delta = np.nanmax([max_rel_e, max_rel_rho, max_rel_rhou, max_rel_rhow, max_rel_x, max_rel_y])

        self.dt = p / self.delta
        if self.dt > 0.1:
            self.dt =  0.1
        elif self.dt < 1e-3:
            self.dt = 1e-3

    def boundary_conditions(self):
        """
        Boundary conditions for energy, density and velocity
        """
        self.alpha_top = self.g * self.mu * self.m_u / (self.kb * self.T[-1, :])
        self.alpha_bottom = self.g * self.mu * self.m_u / (self.kb * self.T[0, :])

        self.e[-1, :] = (4 * self.e[-2, :] - self.e[-3, :]) / (3 + 2*self.dy * self.alpha_top)
        self.e[0, :] = (4 * self.e[1, :] - self.e[2, :]) / (3 - 2*self.dy * self.alpha_bottom)

        self.rho[-1, :] = (4*self.rho[-2, :] - self.rho[-3, :])/(3 + 2*self.dy*self.alpha_top)
        self.rho[0, :] = (4*self.rho[1, :] - self.rho[2, :])/(3 - 2*self.dy*self.alpha_bottom)
        
        self.u[-1, :] = (4*self.u[-2,:] - self.u[-3,:])/3
        self.u[0, :] = (4*self.u[ 1,:] - self.u[ 2,:])/3
        
        self.w[-1, :] = 0
        self.w[0, :] = 0

    def central_x(self, var:np.ndarray):
        """ 
        Central difference scheme in x-direction

        Parameters
        ----------
        var : array-like
            Variable to calculate the central difference of
        
        Returns
        -------
        central_diff : array-like
            Central difference of the variable in x-direction
        """
        var_right = np.roll(var, -1, axis=1)
        var_left = np.roll(var, 1, axis=1)
        central_diff = (var_right - var_left) / (2 * self.dx)
        return central_diff

    def central_y(self, var:np.ndarray):
        """
        Central difference scheme in y-direction
        
        Parameters
        ----------
        var : array-like
            Variable to calculate the central difference of
        
        Returns
        -------
        central_diff : array-like
            Central difference of the variable in y-direction
        """
        var_up = np.roll(var, -1, axis=0)
        var_down = np.roll(var, 1, axis=0)
        central_diff = (var_up - var_down) / (2 * self.dy)
        return central_diff

    def upwind_x(self, var:np.ndarray, v:np.ndarray):
        """
        Upwind difference scheme in x-direction

        Parameters
        ----------
        var : array-like
            Variable to calculate the upwind difference of
        v : array-like
            Velocity in x-direction

        Returns
        -------
        upwind_diff : array-like
            Upwind difference of the variable in x-direction
        """
        var_right = np.roll(var, -1, axis=1)
        var_left = np.roll(var, 1, axis=1)

        pos_vel = (var - var_left) / self.dx
        neg_vel = (var_right - var) / self.dx

        upwind_diff = np.where(v<0, neg_vel, pos_vel)
        return upwind_diff

    def upwind_y(self, var:np.ndarray, v:np.ndarray):
        """
        Upwind difference scheme in y-direction
        
        Parameters
        ----------
        var : array-like
            Variable to calculate the upwind difference of
        v : array-like
            Velocity in y-direction
            
        Returns
        -------
        upwind_diff : array-like
            Upwind difference of the variable in y-direction
        """
        var_up = np.roll(var, -1, axis=0)
        var_down = np.roll(var, 1, axis=0)

        pos_vel = (var - var_down) / self.dy
        neg_vel = (var_up - var) / self.dy

        upwind_diff = np.where(v<0, neg_vel, pos_vel)
        return upwind_diff

    def hydro_solver(self):
        """
        Hydrodynamic equations solver
        """
        self.get_dedt(self.u, self.w, self.e)
        self.get_drhodt(self.u, self.w, self.rho)
        self.get_drhoudt(self.u, self.w, self.rho, self.P)
        self.get_drhowdt(self.u, self.w, self.rho, self.P)
        
        self.timestep()

        self.e[:,:] = self.e + self.dedt * self.dt
        self.rho[:,:] = self.rho + self.drhodt * self.dt
        self.u[:,:] = (self.rho * self.u + self.drhoudt * self.dt) / self.rho
        self.w[:,:] = (self.rho * self.w + self.drhowdt * self.dt) / self.rho

        self.boundary_conditions()

        self.P[:,:] = (self.gamma - 1) * self.e
        self.T[:,:] = (self.gamma - 1) * (self.mu * self.m_u) / (self.rho * self.kb) * self.e

        dt = self.dt
        return dt

def sanity_check():
    """
    Sanity check for the hydro solver
    """
    vis = FVis.FluidVisualiser()
    solver = convection2D(2/5)
    solver.initialise()
    vis.save_data(60, solver.hydro_solver, rho=solver.rho, e=solver.e, u=solver.u, w=solver.w, T=solver.T, P=solver.P, folder='sanity_check')
    vis.animate_2D('w', folder='sanity_check', save=True, video_fps=24, video_name='sanity_check_w')
    vis.animate_2D('T', folder='sanity_check', save=True, video_fps=24, video_name='sanity_check_T')
if check_sanity:
    sanity_check()

def visualise(nabla:float, runtime:int, name:str, variables:list, perturbations:int = 0, save_data:bool=False, save_videos:bool=False):
    """
    Visualise the result of the hydro solver for a given nabla, number of perturbations and runtime.

    Parameters
    ----------
    nabla : float
        Temperature gradient
    runtime : int
        Runtime of the hydro solver in seconds
    name : str
        Name of the folder to save the data and the video
    perturbations : int, optional
        Number of perturbations, 0, 1, 3, and 5 are valid options, 0 is the default
    """
    vis = FVis.FluidVisualiser()
    if save_data:
        solver = convection2D(nabla, perturbations)
        solver.initialise()
        solver.hydro_solver()
        vis.save_data(runtime, solver.hydro_solver, rho=solver.rho, e=solver.e, u=solver.u, w=solver.w, T=solver.T, P=solver.P, folder=name)
    if save_videos:
        for var in variables:
            vis.animate_2D(var, folder = name, save=True, video_name=name+'_'+var, video_fps=24)

# visualise(2/5, 1000, '5_perturbations_0.2_nabla_1000', ['T', 'w', 'e'], 5, save_data=True, save_videos=True)
# visualise(1/2, 1000, '5_perturbations_0.5_nabla_1000', ['T', 'w', 'e'], 5, save_data=True, save_videos=True)
visualise(5, 1000, '5_perturbations_5_nabla_1000', ['T', 'w', 'e'], 5, save_data=True, save_videos=True)
# visualise(9, 1000, '5_perturbations_9_nabla_1000', ['T', 'w', 'e'], 5, save_data=True, save_videos=True)