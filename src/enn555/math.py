import numpy as np

def sind(x):
    return np.sin(np.deg2rad(x))

def cosd(x): 
    return np.cos(np.deg2rad(x))

def tand(x):
    return np.tan(np.deg2rad(x))

def acosd(x):
    return np.rad2deg( np.arccos(x) )

def asind(x):
    return np.rad2deg( np.arcsin(x) )

def atand(x):
    return np.rad2deg( np.arctan(x) )

def vecnorm(x):
    return np.sqrt(np.dot(x,x))

def proj(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/vecnorm(b) * b