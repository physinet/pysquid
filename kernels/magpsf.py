"""
magpsf.py

    author: Colin Clement
    email: colin.clement@gmail.com
    date: 2015-9-10

This contains classes which contain kernels with the Biot-Savart law
information and a point-spread function (PSF).

"""

from __future__ import division, print_function
import numpy as np
from pysquid.kernels.kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self, shape, psf_params = None,
                 padding = None, **kwargs):
        """
        Single Gaussian PSF kernel. params are x and y-standard deviations
        input:
            shape : tuple of ints (ly, lx),  shape of measured flux field
            psf_params : 3-element numpy array.  list of parameters for 
                    point spread function. 
                    First element is always height above plane. Second and third
                    elements are sigma_x and sigma_y of Gaussian
            padding : tuple of ints (py, px) for padding on g-field outside
                      flux image

        kwargs:
            dx and dy: floats for units of x and y. Choose either dx or dy
            to be 1, and then the other should be the appropriate ratio.
            NOTE: You should choose the smaller of x and y units to be 1,
            so that the larger is a number greater than 1, which will promote
            numerical stability in the linear solvers.
            
            _fftw_flags and _fftw_threads (see FFTW object)


        """
        self.psf_params = psf_params if psf_params is not None else np.array([1., 1., 1.])
        super(GaussianKernel, self).__init__(shape, self.psf_params, 
                                             padding, **kwargs)
        if not len(self.psf_params) == 3:
            raise RuntimeError('len(params) must be 3')
         
        self._updatePSF()
        self._updateMPSF()

    def _updatePSF(self):
        z, sx, sy = self.psf_params
        #Shifted to center at 0,0 (to match fourier method)
        x0, y0 = self.d_xg[0,0], self.d_yg[0,0]
        d_xg, d_yg = self.d_xg - x0, self.d_yg - y0 
        self.PSF = np.exp(-((d_xg/sx)**2 +
                            (d_yg/sy)**2 )/2)#/(2*np.pi*sx*sy)
        self.PSF /= self.PSF.sum()*self.dx*self.dy
        self.d_PSF = np.rollaxis(np.array([self.PSF*((d_xg**2-sx*sx)/sx**3),
                                           self.PSF*((d_yg**2-sy*sy)/sy**3)]),
                                 0, 3) #Shape (ly, lx, N_psf_params)

        self.d_PSF_k = np.rollaxis(np.array([self.fftw.fft(self.d_PSF[:,:,i]) for i in
                                             range(self.d_PSF.shape[2])]), 0, 3)
        self.PSF_k = self.fftw.fft(self.PSF)


class MogKernel(Kernel):
    def __init__(self, shape, psf_params, padding = None, **kwargs):
        """
        Mixture of Gaussians PSF kernel.
        input:
            shape : tuple of ints (ly, lx),  shape of measured flux field
            psf_params : list of parameters for 
                    point spread function. 
                    First element is always height above plane. What follows
                    is 5n parameters for n gaussians. the gaussian
                    parameters are (amplitude, mean_x, mean_y, sigma_x, sigma_y)
            padding : tuple of ints (py, px) for padding on g-field outside
                      flux image

        kwargs:
            dx and dy: floats for units of x and y. Choose either dx or dy
            to be 1, and then the other should be the appropriate ratio.
            NOTE: You should choose the smaller of x and y units to be 1,
            so that the larger is a number greater than 1, which will promote
            numerical stability in the linear solvers.
            
            _fftw_flags and _fftw_threads (see FFTW object)

        """
        self.psf_params = psf_params.ravel()
        if (self.psf_params.size - 1) % 5:
            raise RuntimeError("psf_params must be of size 5n+1")
        super(MogKernel, self).__init__(shape, self.psf_params, padding,
                                        **kwargs)
        self.N_g = (self.psf_params.size-1) // 5
        self._updatePSF()
        self._updateMPSF()

    def _updatePSF(self):
        #Log params for a, sx, sy
        p = self.psf_params[1:].reshape(self.N_g, 5)
        x, y, sx, sy = p[:,1], p[:,2], p[:,3], p[:,4]
        xg, yg = self.d_xg - self.d_xg[0,0], self.d_yg - self.d_yg[0,0]
        Dx, Dy = xg[:,:,None]-x[None,None,:], yg[:,:,None]-y[None,None,:]
        g = np.exp(- Dx**2/(2*sx[None,None,:]**2) - Dy**2/(2*sy[None,None,:]**2))
        g = p[:,0][None,None,:]*g/np.sqrt(2*np.pi*sx*sy[None,None,:])
        self.PSF = g.sum(2)
        self.PSF /= self.PSF.sum()*self.dx*self.dy
        self.PSF_k = self.fftw.fft(self.PSF)
        # Gradient not implemented
        self.d_PSF_k = np.zeros((2*self.Ly_pad, 2*self.Lx_pad, 5*self.N_g))



