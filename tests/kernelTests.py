import numpy as np
from pysquid.kernels.magpsf import *
from pysquid.kernels.psf import *
from pysquid.util.helpers import makeM

def testConvolution(kernelType, psf_params = None, shape = (10, 10), 
                    padding = (2, 2), gtest = None, **kwargs):
    """
    This function tests the validity of using a Fourier Transform with a cutoff
    kernel to perform the convolution of the Biot-Savart law and PSF.

    """
    kernel = kernelType(shape, psf_params, padding, cutoff = False, **kwargs)
    MPSF = np.fft.fftshift(kernel.MPSF.copy()).real
    M = makeM(MPSF, padding)
    gtest = gtest if gtest is not None else np.random.randn(*kernel._padshape)
    fftConv = kernel.applyM(gtest).real
    trueConv = M.dot(gtest.ravel()).reshape(*shape)
    rmserror = np.sqrt(np.mean((fftConv - trueConv)**2))
    return rmserror, kernel, M


