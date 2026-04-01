import numpy as np
from numpy.typing import NDArray
from typing import Callable,Tuple,List
import matplotlib.pyplot as plt


class Particle:
    def __init__(self,i,x,y,
                theta,radius) -> None:
        self.id = i
        # wrapped (for simulation)
        self.x = x
        self.y = y
        
        # unwrapped (for tracking)
        self.x_unwrapped = x
        self.y_unwrapped = y
        
        self.theta = theta
        self.radius = radius
        self.v = 5.0

def populate_is_occupied(radius,xc,yc,id,is_occupied: NDArray,
                        collisions:List[Tuple[int,int]]):
    Ny,Nx = is_occupied.shape
    
    xa = max(0,int(xc-radius))
    xb = min(Nx , int(xc+radius)+1)
    ya = max(0,int(yc-radius))
    yb = min(Ny , int(yc+radius)+1)
    
    for x in range(xa,xb):
        for y in range(ya,yb):    
            if (x-xc)**2 + (y-yc)**2 <= radius*radius:
                xp = x % Nx
                yp = y % Ny
                if is_occupied[yp,xp] == -1:
                    is_occupied[yp,xp] = id
                else:
                    # so we will define some function like 
                    # def collision
                    collisions.append((id,is_occupied[yp,xp]))


def move(particle:Particle,Nx:int,Ny:int)->None:
    
    # compute displacement
    dx = particle.v * np.cos(particle.theta)
    dy = particle.v * np.sin(particle.theta)
    
    # update wrapped position (for simulation)
    particle.x = (particle.x + dx) % Nx
    particle.y = (particle.y + dy) % Ny
    
    # update unwrapped position (NO modulo!)
    particle.x_unwrapped += dx
    particle.y_unwrapped += dy
    

def deposit_chemB(particles: List[Particle],chemB: NDArray,strength : float = 1.0):
    
    Ny,Nx = chemB.shape
    for particle in particles:
        xc = particle.x 
        yc = particle.y
        r = particle.radius
        
        xa = int(xc-r)
        xb = int(xc+r)+1
        ya = int(yc-r)
        yb = int(yc+r)+1
    
        for x in range(xa,xb):
            for y in range(ya,yb):
                if (x-xc)**2 + (y-yc)**2 <= r*r:
                    xp = x%Nx
                    yp = y%Ny
                    chemB[yp,xp] += strength

def laplacian(field:NDArray)-> NDArray:
    lap_field = np.full(shape=field.shape,fill_value=np.nan)
    Ny,Nx = field.shape
    
    for x in range(Nx):
        for y in range(Ny):
            
            term = (field[(y+1)%Ny,x] + field[(y-1)%Ny,x] +
                    field[y,(x+1)%Nx] + field[y,(x-1)%Nx] -
                    4 * field[y,x])
            lap_field[y,x] = term
            
    return lap_field    
def init_gaussian(chemB, x0, y0, sigma=10.0, amplitude=1.0):
    Ny, Nx = chemB.shape
    
    for y in range(Ny):
        for x in range(Nx):
            dx = x - x0
            dy = y - y0
            
            chemB[y, x] += amplitude * np.exp(
                -(dx*dx + dy*dy) / (2*sigma*sigma))


def populate(particles:List[Particle],is_occupied:NDArray,
            chemA:NDArray,chemB:NDArray)->Tuple[List[Particle],NDArray,NDArray,NDArray]:
# will populate and initialize system
    return (particles,is_occupied,chemA,chemB)
    
