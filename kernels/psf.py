"""
psf.py

    author: Colin Clement
    email: colin.clement@gmail.com
    date: 2015-9-10

This contains classes which describe kernels without the Biot-Savart law
and ONLY with a point-spread-function (PSF)

"""

from __future__ import division, print_function
import numpy as np
from pysquid.kernels.kernel import Kernel


class PSFKernel(Kernel):

    def _updateM(self): #No g-kernel
        pass #Leave alone

    def _updateMPSF(self): #Just PSF
        self.MPSF_k = self.PSF_k 
        self.MPSF = self.fftw.ifft(self.MPSF_k)

    def _updatePSF(self):
        pass #implement PSF

class GaussianBlurKernel(PSFKernel):
    def __init__(self, shape, psf_params = None,
                 padding = None, **kwargs):
        """
        Single Gaussian PSF kernel. params are x and y-standard deviations
        input:
            shape : tuple of ints (ly, lx),  shape of measured flux field
            psf_params : 2-element numpy array.  list of parameters for 
                    point spread function. 
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
        self.psf_params = psf_params if psf_params is not None else np.array([1., 1.])
        super(GaussianBlurKernel, self).__init__(shape, self.psf_params, 
                                                 padding, **kwargs)
        if not len(self.psf_params) == 2:
            raise RuntimeError('len(params) must be 2')
         
        self._updatePSF()
        self._updateMPSF()

    def _updatePSF(self):
        sx, sy = self.psf_params
        #Shifted to center at 0,0 (to match fourier method)
        x0, y0 = self.d_xg[0,0], self.d_yg[0,0]
        d_xg, d_yg = self.d_xg - x0, self.d_yg - y0 
        self.PSF = np.exp(-((d_xg/sx)**2 +
                            (d_yg/sy)**2 )/2)#/(2*np.pi*sx*sy)
        self.PSF /= self.PSF.sum()
        self.d_PSF = np.rollaxis(np.array([self.PSF*((d_xg**2-sx*sx)/sx**3),
                                           self.PSF*((d_yg**2-sy*sy)/sy**3)]),
                                 0, 3) #Shape (ly, lx, N_psf_params)

        self.d_PSF_k = np.rollaxis(np.array([self.fftw.fft(self.d_PSF[:,:,i]) for i in
                                             range(self.d_PSF.shape[2])]), 0, 3)
        self.PSF_k = self.fftw.fft(self.PSF)




