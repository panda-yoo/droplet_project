
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np


@dataclass(slots=True)
class Particle:
    """Minimal active particle used by the notebook simulations."""

    id: int
    x: float
    y: float
    theta: float
    speed: float = 0.1
    radius: float = 0.5
    alpha: float = 0.5
    align_strength: float = 1.0
    Dr: float = 0.1
    noise_amp: float = 0.2
    omega: float = 0.0
    vx: float = field(init=False)
    vy: float = field(init=False)

    def __post_init__(self) -> None:
        self._refresh_velocity()

    def _refresh_velocity(self) -> None:
        self.vx = np.cos(self.theta)
        self.vy = np.sin(self.theta)

    def update_theta(self, alignment: float, gradx: float, grady: float) -> None:
        """Rotate the particle by combining alignment, chemotaxis, and noise."""

        noise = np.random.normal(scale=self.noise_amp)

        align_term = self.align_strength * (alignment - self.theta)
        chem_term = self.alpha * (
            gradx * (-np.sin(self.theta)) +
            grady * np.cos(self.theta)
        )
        noise_term = np.sqrt(2 * self.Dr) * noise

        self.theta = (self.theta + align_term + chem_term + noise_term) % (2 * np.pi)
        self._refresh_velocity()

    def move(self, L: float) -> None:
        """Advance the particle by one Euler-Maruyama step with periodic wrap."""

        self.x = (self.x + self.speed * self.vx) % L
        self.y = (self.y + self.speed * self.vy) % L


def compute_alignment(i: int, particles: Sequence[Particle], R: float, L: float) -> float:
    """Return the local mean direction seen by particle i (Vicsek-style)."""

    px, py = particles[i].x, particles[i].y
    sumx = 0.0
    sumy = 0.0

    for j, pj in enumerate(particles):
        dx = px - pj.x
        dy = py - pj.y

        if dx > L / 2:
            dx -= L
        elif dx < -L / 2:
            dx += L

        if dy > L / 2:
            dy -= L
        elif dy < -L / 2:
            dy += L

        if dx * dx + dy * dy < R * R:
            sumx += np.cos(pj.theta)
            sumy += np.sin(pj.theta)

    if sumx == 0.0 and sumy == 0.0:
        return particles[i].theta

    return np.arctan2(sumy, sumx)


def compute_gradient(p: Particle, C: np.ndarray, Nx: int, Ny: int, L: float) -> Tuple[float, float]:
    """Centered finite-difference gradient of scalar field C at particle p."""

    ix = int(p.x / L * Nx)
    iy = int(p.y / L * Ny)

    ix = max(1, min(ix, Nx - 2))
    iy = max(1, min(iy, Ny - 2))

    gradx = 0.5 * (C[ix + 1, iy] - C[ix - 1, iy])
    grady = 0.5 * (C[ix, iy + 1] - C[ix, iy - 1])

    return gradx, grady


"""       
class Particle:
    def __init__(self,id:int) -> None:
        # Unique ID
        self.id = id
        # --- position --- 
        self.x = None
        self.y = None
        # --- velocity ---
        self.vx = 0.0
        self.vy = 0.0
        # --- Orientation ---
        self.theta = None
        self.omega = None
        # --- Interaction strengths ---
        self.alpha = None
        self.align_strength = None
        # --- Noise / persistence ---
        self.Dr = None
        self.noise_amp = None
        
    
    def update_position(self,x:float,y:float)-> None:
        self.x = x
        self.y = y
        
    def update_velocity(self,vx:float,vy:float)-> None:
        self.vx = vx
        self.vy = vy
        
    def update_orientation(self,theta:float,omega:float)-> None:
        self.theta = theta
        self.omega = omega
    
    def update_Interaction_strengths(self,alpha:float,align_strength:float)-> None:
        self.alpha = alpha
        self.align_strength = align_strength
    
    def update_Noise(self,Dr:float,noise_amp:float)-> None:
        self.Dr = Dr
        self.noise_amp = noise_amp
    
    
    # initialize
    def initialize(self,x: float,y: float,vx: float,vy: float,
                   theta: float,omega: float,alpha: float,align_strength: float,
                   Dr:float,noise_amp:float)->None:
        
        self.update_position(x,y)
        self.update_velocity(vx,vy)
        self.update_orientation(theta,omega)
        self.update_Interaction_strengths(alpha,align_strength)
        self.update_Noise(Dr,noise_amp)
                        
"""