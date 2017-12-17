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
        assert len(self.psf_params) == 2, 'len(params) must be 2'
         
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


from scipy.special import expit, logit


class PlatonicKernel(PSFKernel):
    def __init__(self, shape, psf_params = None,
                 padding = None, **kwargs):
        """
        This is an implementation of a platonic approximation
        to the PSF of a key-shaped squid-loop. It is a solid,
        smoothed circle and rectangle with reduced amplitude.
        input:
            shape: tuple (Ly, Lx) of data shape
            psf_params: ndarray of log-parameters
                        [radius, length, width, keyamp, amp]

            NOTE: this model PSF transforms parameters to enforce
            a number of conditions. radius and length are made positive
            by using logarithmic parameters, i.e. the model
            takes their exponential. width and keyamp are restricted
            to be in the range (0,1) by taking the expit. width
            is then understood as fraction of the diameter (2*radius).
            amp is enforced to be positive. Use the methods
            self.get_p to transform to the constraint-satisfying
            parameters, and self.get_logp to transform to the
            unconstrained parameters.

            **kwargs:
                d: transition rate for sigmoidal smoothing of 
                    the platonic shapes
                h: step-size for central difference derivatives
                    of PSF with respect to parameters
        """
        default = np.log(np.array([5., 5., .7, 0.33, 1.]))
        self.d = kwargs.get('d', 0.25)
        self.h = kwargs.get('h', 1E-7)
        self.psf_params = psf_params if psf_params is not None else default
        super(PlatonicKernel, self).__init__(shape, self.psf_params, 
                                             padding, **kwargs)
        assert len(self.psf_params) == 5, "len(params) must be 5"

        d_psf_shape = (2*self.Ly_pad, 2*self.Lx_pad, len(self.psf_params))
        self.d_PSF = np.zeros(d_psf_shape)
        self.d_PSF_k = np.zeros(d_psf_shape, dtype='complex128')
        self._updatePSF()
        self._updateMPSF()

    def _updatePSF(self):
        p0 = self.psf_params.copy()
        self.PSF = self._psf(p0) 
        self.PSF_k = self.fftw.fft(self.PSF)

        for ip in range(len(p0)):
            p0[ip] += self.h/2.
            plus_h = self._psf(p0)
            p0[ip] -= self.h
            minus_h = self._psf(p0)
            self.d_PSF[:,:,ip] = (plus_h - minus_h)/self.h
            p0[ip] += self.h/2.
            self.d_PSF_k[:,:,ip] = self.fftw.fft(self.d_PSF[:,:,ip])

    @staticmethod
    def get_logp(params):
        """
        transform params back to unconstrained domains
        """
        logp = np.log(params)
        p2, p3 = params[2:4]
        logp[2], logp[3] = logit(params[2:4])
        return logp

    @staticmethod
    def get_p(params):
        """
        transform params into the constrained domains
        """
        p = np.exp(params)
        p[2:4] = expit(params[2:4])
        return p

    @property
    def p(self):
        """
        Get the constrained parameters
        """
        return self.get_p(self.psf_params)

    def _circle(self, r):
        """
        Make image of smoothed circle with radius r
        """
        x0, y0 = self.d_xg[0,0], self.d_yg[0,0]
        d_xg, d_yg = self.d_xg - x0, self.d_yg - y0 
        radial = np.hypot(d_xg, d_yg)
        return expit(-(radial - r)/self.d)
    
    def _rectangle(self, L, w, offset):
        """
        Make smoothed rectangle of height (length) L,
        width w, offset from the origin by offset.
        """
        x0, y0 = self.d_xg[0,0], self.d_yg[0,0] - offset
        d_xg, d_yg = self.d_xg - x0, self.d_yg - y0
        sigx = expit(-(np.abs(d_xg) - w/2.)/self.d)
        sigy = expit(-(np.abs(d_yg + L/2) - L/2)/self.d)
        return sigx*sigy

    def _psf(self, params):
        """
        Make the platonic keyhole-shaped psf with
        unconstrained parameters params
        """
        radius, length, width, kamp, amp = self.get_p(params)
        circ = self._circle(radius)
        offset = radius*np.sqrt(1 - width**2)
        rect = self._rectangle(length, width*2*radius, offset)
        out = np.maximum(circ, kamp * rect)
        return out*amp

