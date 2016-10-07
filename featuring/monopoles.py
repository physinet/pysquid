"""
monopoles.py

author: Colin Clement
date: 2016-09-21


"""

import numpy as np

from pysquid.opt import leastsq
from pysquid.parametermap import ParameterMap
from pysquid.kernels import psf
from pysquid.component import ModelComponent
from pysquid.featuring.fields import monopoleField, grad_monopoleField


class FeatureMonopoles(ModelComponent):
    def __init__(self, shape, N_m, kernel = None, coords = None, 
                 z = None, psf_params = None, padding = None, **kwargs):

        super(FeatureMonopoles, self).__init__(shape, padding, **kwargs)

        self.N_m = N_m
        self.z = z if z is not None else 5*np.random.rand()
        
        defaultc = np.c_[self.Lx*np.random.rand(N_m), #x,y coords
                         self.Ly*np.random.rand(N_m)].ravel()
        self.coords = coords if coords is not None else defaultc
        assert len(self.coords) == 2*N_m, "len(coords) must be 2*N_m"

        if kernel is not None:
            assert issubclass(kernel, psf.PSFKernel), \
                    "kernel must be a PSFKernel"
            self.kernel = kernel(shape, psf_params, padding, **kwargs)
        else:
            self.kernel = psf.PlatonicKernel(shape, psf_params, padding,
                                             **kwargs)

        self.paramsizes = {'z': 1, 'coords': 2*self.N_m, 
                           'psf_params': len(self.kernel.psf_params)}

        self.N_psfparams = len(self.kernel.psf_params)
        self.N_params = 1+len(self.coords)+self.N_psfparams
        self.field = np.zeros((self.Ly_pad, self.Lx_pad))
        self.gradflux = np.zeros((self.N_pad, self.N_params))

        self._updateGrids()

    def _updateGrids(self):
        self.x = self.dx * np.arange( 0.5,  self.Lx_pad,  1.)
        self.y = self.dy * np.arange(-0.5, -self.Ly_pad, -1.)
        self.xg, self.yg = np.meshgrid(self.x, self.y)
        self.d_xg = self.xg - (self.dx*self.Lx_pad)/2
        self.d_yg = self.yg + (self.dy*self.Ly_pad)/2

    def updateParams(self, name, values):
        if name == 'psf_params':
            self.kernel.updateParams(name, values)
        elif name == 'coords':
            self.coords = values
            self.updateField()
        elif name == 'z':
            self.z = values[0]
            self.updateField()
        self.updateFlux()

    def updateParamVector(self, names, values):
        count = 0
        for name in names:
            size = self.paramsizes[name]
            self.updateParams(name, values[count:count+size])
            count += size

    def updateField(self):
        self.field.fill(0.)
        for cc in self.coords.reshape(-1,2):
            self.field += monopoleField(self.z, cc[0], cc[1], 
                                        self.d_xg, self.d_yg)

    def updateFlux(self):
        self.flux = self.kernel.applyM(self.field).real.ravel()

    def updateGradFlux(self):
        self.gradflux[:,0] *= 0. #df/dz
        applyM = self.kernel.applyM
        for ic, cc in enumerate(self.coords.reshape(-1,2)):
            gradfield = grad_monopoleField(self.z, cc[0], cc[1],
                                           self.d_xg, self.d_yg)
            self.gradflux[:,0] += gradfield[0,:,:].ravel()
            self.gradflux[:,2*ic+1] = applyM(gradfield[1,:,:]).real.ravel()
            self.gradflux[:,2*ic+2] = applyM(gradfield[2,:,:]).real.ravel()
        dfield_dz = self.gradflux[:,0].reshape(self.Ly_pad,-1)
        self.gradflux[:,0] = applyM(dfield_dz).real.ravel()
        #self.updateField()
        grad_psf = self.kernel.computeGradients(self.field)
        self.gradflux[:,1+2*self.N_m:] = grad_psf.reshape(-1,self.N_psfparams)

    def residual(self, paramvec, names, data):
        self.updateParamVector(names, paramvec)
        return self.flux - data.ravel()

    def grad_residual(self, paramvec, names, data):
        self.updateParamVector(names, paramvec)
        self.updateGradFlux()
        return self.gradflux

    def optimize(self, p0, names, data, **kwargs):
        mylm = leastsq.LM(self.N, len(p0),
                          self.residual, self.grad_residual,
                          args = (names, data))
        self.soln = mylm.leastsq(p0, **kwargs)
        return self.soln[0]
         
        
         

