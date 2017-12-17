"""
fields.py

author: Colin Clement
date: 2016-09-21

This script compute the perpendicular magnetic field due to a magnetic
monopole and derivatives with respect to its parameterization.
"""


import numpy as np
import numexpr as nu

def monopoleField(z, x0, y0, x, y):
    """
    Magnetic field of a flux vortex for which only one
    end is 'nearby', so that the field is essentially a monopole.
    input:
        z : (float) height of measurement plane above currents
        x0, y0 : (floats) coordinates of the center of the monopole
        x, y: float arrays of equal shape giving the points at which
                the flux is to be evaluated.
    returns:
        float array the same shape as (x0, y0) of the field of a 
        monopole in units of \Phi_0 the flux quantum.
    """
    xc, yc = x - x0, y - y0
    pi = np.pi
    return nu.evaluate("1./(2*pi*(xc*xc+ yc*yc + z*z))")

def grad_monopoleField(z, x0, y0, x, y):
    """
    Derivative of magnetic monopole field with respect to x0, y0, z.
    input:
        same as monopoleField
    returns:
        ndarray [grad_x0, grad_y0, grad_z] where each grad
        is the shape of the output from monopoleField
    """
    B = monopoleField(z, x0, y0, x, y)
    B2 = 4*np.pi*B*B
    return np.array([-z*B2, (x-x0)*B2, (y-y0)*B2])
