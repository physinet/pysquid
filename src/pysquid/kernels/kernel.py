"""
kernel.py

    author: Colin Clement
    email: colin.clement@gmail.com
    date: 2015-9-10

This is a class for managing the construction of the M matrix for computing
magnetic flux due to g-fields

"""


from __future__ import division, print_function
import numpy as np
import numexpr as nu
from numpy.fft import fftshift
from copy import copy
from pysquid.component import ModelComponent
from pysquid.util.fftw import WrapFFTW
from pysquid.util.helpers import _mult


class Kernel(ModelComponent):
    
    def __init__(self, shape, params = None, padding = None,
                 **kwargs):
        """
        Kernel class instance stores point spread functions and the flux of
        point (square) sources of constant g-field. 

        input:
            shape : tuple of ints (ly, lx),  shape of measured flux field
            params : list of parameters for point spread function. 
                     First element is always height above plane.
                     Length is dependent on subclass implementations
            padding : tuple of ints (py, px) for padding on g-field outside
                      flux image

        kwargs:
            dx and dy: floats for units of x and y. Choose either dx or dy
            to be 1, and then the other should be the appropriate ratio.
            NOTE: You should choose the smaller of x and y units to be 1,
            so that the larger is a number greater than 1, which will promote
            numerical stability in the linear solvers.
            
            _fftw_flags and _fftw_threads (see FFTW object)

            cutoff: Default is true, will cutoff PSF.
        """
        super(Kernel, self).__init__(shape, padding, **kwargs)

        self.PSF = np.zeros(self._fftshape)
        self.PSF[0,0] = 1.
        
        self._fftw_flags = kwargs.get('_fftw_flags', ['FFTW_MEASURE'])
        self._fftw_threads = kwargs.get('_fftw_threads', 8)

        self._updateGrids()

        self.fftw = WrapFFTW(self._fftshape, flags = self._fftw_flags,
                             threads = self._fftw_threads)
        self._unPickleable += ['fftw']

        # Workspace arrays
        self._padExpandG = np.zeros(self._fftshape)
        self._padFlux = np.zeros_like(self._padExpandG)
        self._conv_k = np.zeros(self._fftshape, dtype='complex128')
        self._dMPSF_dz_k = np.zeros_like(self._conv_k)
        self._Mtoutput = np.zeros((self.Ly_pad, self.Lx_pad))
        self.dMPSF_dz_dotg = np.zeros((self.Ly, self.Lx, len(self.psf_params)))

        self._updateM()

    def __setstate__(self, d):
        self.__dict__ = d
        self.fftw = WrapFFTW(self._fftshape, flags = self._fftw_flags,
                             threads = self._fftw_threads)

    def updateParams(self, name, values):
        """
        params = list of at least one number
        set up for lazy evaluation
        """
        if name == 'psf_params':
            self.psf_params = copy(values)
            self._updateM()
            self._updatePSF()
            self._updateMPSF()

    def _updateGrids(self):
        self.x = self.dx * np.arange( 0.5,  2*self.Lx_pad,  1.)
        self.y = self.dy * np.arange(-0.5, -2*self.Ly_pad, -1.)
        self.xg, self.yg = np.meshgrid(self.x, self.y)
        self.d_xg = fftshift(self.xg - (self.dx*2*self.Lx_pad)/2)
        self.d_yg = fftshift(self.yg + (self.dy*2*self.Ly_pad)/2)

    def _updateM(self):
        self.M_g, self.dM_g_dz = _gGreensFunction(self.dx/2., -self.dy/2., 
                                                  self.d_xg, self.d_yg,
                                                  self.psf_params[0], 
                                                  self.dx, self.dy)

    def _updatePSF(self):
        """
        Should put psf in self.PSF, its forier transform in
        self.PSF_k, dpsf_dz in self.d_PSF
        and the fourier transform of dpsf_dz in d_PSF_k
        where z are the psf_params
        self.d_PSF should be shape (ly, lx, N_params)
        where N_params is the number of PSF parameters

        """
        pass

    def _updateMPSF(self):
        self.M_g_k = self.fftw.fft(self.M_g)
        self.MPSF_k = self.M_g_k * self.PSF_k
        self.MPSF = self.fftw.ifft(self.MPSF_k)

    def convolveWithPSF(self, arr):
        """
        Convolve arr with PSF of this kernel instance
        """
        _mult(self._conv_k, self.fftw.fft(arr), self.PSF_k)
        return self.fftw.ifft(self._conv_k)

    def _apply(self, M_k, expanded_arr):
        """
        Convolve arr (real space) with M_k (kernel in k-space)
        """
        _mult(self._conv_k, self.fftw.fft(expanded_arr), M_k)
        return self.fftw.ifft(self._conv_k)

    def applyM(self, arr):
        """
        Compute product M.dot(g) using FFT.
        input:
            arr : (g) (Ly_pad, Lx_pad)-shaped or (N_pad)-shaped float array
        output:
            M.dot(arr) : (Ly_pad, Lx_pad)-shaped array
        """
        Ly_pad, Lx_pad = self.Ly_pad, self.Lx_pad
        self._padExpandG[:Ly_pad, :Lx_pad] = arr.reshape(Ly_pad, Lx_pad)[:,:]
        return self.crop(self._apply(self.MPSF_k, self._padExpandG)[:Ly_pad,
                                                                    :Lx_pad])

    def applyMt(self, arr):
        """
        Compute product M.T.dot(arr) using FFT.
        input:
            arr : (flux) (Ly, Lx)-shaped or (Ly*Lx)-shaped float array
        output:
            M.T.dot(arr) : (Ly_pad,Lx_pad)-shaped array where, e.g.
        """
        Ly, Lx, py, px = self.Ly, self.Lx, self.py, self.px
        self._padFlux[py:py+Ly,px:px+Lx] = arr.reshape(Ly, Lx)
        self._flux_transpose = self._apply(self.MPSF_k, self._padFlux)
        return self._flux_transpose[:self.Ly_pad, :self.Lx_pad]

    def computeGradients(self, arr):
        """
        Compute product of dM_dz.dot(arr) using FFT.
        input:
            arr: (g) (Ly, Lx)-shaped or (Ly*Lx)-shaped float array
        output:
            dM_dz.dot(arr) : (ly, lx, N)-shaped array where
                             N is the number of parameters
        """
        self._padExpandG[:self.Ly_pad, :self.Lx_pad] = arr[:,:]
        self._padExpandG_k = self.fftw.fft(self._padExpandG)
       
        offset = 0
        if hasattr(self, 'dM_g_dz'): 
            _mult(self._dMPSF_dz_k, self.fftw.fft(self.dM_g_dz), self.PSF_k)
            _mult(self._conv_k, self._dMPSF_dz_k, self._padExpandG_k)
            self.dMPSF_dz_dotg[:,:,0] = self.crop(self.fftw.ifft(self._conv_k).real)
            offset = 1
        
        for p in range(len(self.psf_params)-offset):
            if hasattr(self, 'M_g_k'): 
                _mult(self._dMPSF_dz_k, self.d_PSF_k[:,:,p], self.M_g_k)
                _mult(self._conv_k, self._dMPSF_dz_k, self._padExpandG_k)
            else: #Just convolve with PSF
                _mult(self._conv_k, self.d_PSF_k[:,:,p], self._padExpandG_k)

            self.dMPSF_dz_dotg[:,:,p+offset] = self.crop(self.fftw.ifft(self._conv_k).real)        
        return self.dMPSF_dz_dotg

    @property
    def M(self):
        return self.MPSF.real

    @property
    def M_k(self):
        return self.MPSF_k


class BareKernel(Kernel):
    def __init__(self, shape, psf_params = [1.], padding = None, **kwargs):
        """
        A kernel with no point spread function for computing magnetic fields
        """
        self.psf_params = psf_params
        super(BareKernel, self).__init__(shape, psf_params, padding, **kwargs)
        self._updatePSF()
        self._updateMPSF()

    def _updateMPSF(self): #Just Biot-Savart kernel
        self.MPSF = self.M_g
        self.MPSF_k = self.fftw.fft(self.M_g)


################ Bz-field functions for unit g-fields ##################

def _indefIntegral(x,y,d):
    """
    Indefinite integral of g-field flux for thin sheet of current 
    """
    d2, x2, y2 = d*d, x*x, y*y
    return nu.evaluate('x*y*(2*d2+x2+y2)/((d2+x2)*(d2+y2)*sqrt(d2+x2+y2))')

def _d_indefIntegral(x,y,d):
    """
    derivative of Indefinite integral of g-field flux for thin sheet of current 
    with respect to sample plane height d.
    """
    d2, x2, y2 = d*d, x*x, y*y
    Bz = _indefIntegral(x, y, d)
    p = nu.evaluate('d*(4./(2*d2+x2+y2)-1./(d2+x2+y2)-2./(d2+y2)-2./(d2+x2))')
    return Bz * p

def _gGreensFunction(x0, y0, x, y, z, ax, ay):
    """
    Flux field from a square of constant g (square loop of
    constant current), and derivative of said flux with respect to 
    the sample plane height z

    input:
        x0, y0 : (floats) coordinates of the center of the current
        x, y: float arrays of equal shape giving the points at which
                the flux is to be evaluated.
        z : (float) height of measurement plane above currents
        ax, ay : (float) width of current loop

    output:
        phi : array of shape (N, M) where x.shape=y.shape=(N,M),
        dphi_dz : same shape and type as phi
        
    """
    ax2, ay2 = ax/2., ay/2. 
    A = lambda x1, y1: _indefIntegral(x1, y1, z)
    dAdz = lambda x1, y1: _d_indefIntegral(x1, y1, z)
    xc, yc = x - x0, y - y0
    Bz = (A( ax2 - xc, ay2 - yc) - A( ax2 - xc, -ay2 - yc) - 
          A(-ax2 - xc, ay2 - yc) + A(-ax2 - xc, -ay2 - yc))/(4*np.pi)
    dBdz = (dAdz( ax2 - xc, ay2 - yc) - dAdz( ax2 - xc, -ay2 - yc) - 
            dAdz(-ax2 - xc, ay2 - yc) + dAdz(-ax2 - xc, -ay2 - yc))/(4*np.pi)
    return Bz, dBdz

def _lineindefintegral(l, x, y, z):
    b, r = x*x + z*z, l - y
    return x*r/(b*np.sqrt(b+r*r))/(4*np.pi)

def _bzlinecurrent(x, y, x0, y0, z, l, ydir=True):
    """
    Computes the z-component of the magnetic field due to a line of unit current
    of length l, centered at (x0, y0). If ydir is True, the current points in
    the y-direction. otherwise ir points in the x-direction. Sampled at points
    x, y
    """
    if ydir:
        return (_lineindefintegral(l/2., x-x0, y-y0, z) - 
                _lineindefintegral(-l/2., x-x0, y-y0, z))
    else:
        return (_lineindefintegral(l/2., y-y0, x-x0, z) - 
                _lineindefintegral(-l/2., y-y0, x-x0, z))

def _bzfield_edges(gfield, height, rxy=1.):
    Ly, Lx = gfield.shape
    x = np.fft.fftshift(rxy*np.arange(-Lx, Lx)[None,:])
    y = np.fft.fftshift(np.arange(-Ly, Ly)[:,None])
    top = _bzlinecurrent(x, y, 0., -0.5, height, 1., False)
    bottom = - _bzlinecurrent(x, y, 0., 0.5, height, 1., False)
    left = _bzlinecurrent(x, y, -rxy/2., 0., height, rxy)
    right = - _bzlinecurrent(x, y, rxy/2., 0., height, rxy)

    edge = np.zeros((2*Ly, 2*Lx))
    kernels = [top, bottom, left, right]
    slices = [np.s_[0,:Lx], np.s_[Ly-1,:Lx], np.s_[:Ly,0], np.s_[:Ly,Lx-1]]
    edgefield_k = np.zeros((2*Ly, Lx+1), dtype='complex128')
    for k, sl in zip(kernels, slices):
        edges[sl] = gfield[sl].copy()
        edgefield_k += np.fft.rfftn(edge)*np.fft.rfftn(k)
        edge[sl] = 0.
    return np.fft.irfftn(edgefield_k)[:Ly, :Lx]
