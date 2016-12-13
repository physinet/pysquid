"""
tvprior.py

author: Colin Clement
date: 2016-12-06



"""

import scipy.sparse.linalg as ssl
from scipy.optimize import minimize
from scipy.sparse import vstack
import numexpr as nu
import numpy as np

from pysquid.util.linearOperator import MyLinearOperator
from pysquid.util.helpers import makeD2_operators
from pysquid.opt.admm import ADMM


linear_solvers = ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'lgmres', 
                  'minres', 'qmr']
solver_msg = ("Solver must be one of the following\n" + 
              reduce(lambda u,v: u+', '+v, linear_solvers))


class Deconvolver(ADMM):
    """
    Performs deconvolution of flux image with arbitrary
    log-prior. In particular, this sets f(x) in standard
    ADMM problem to be 1/2||Mx-phi||^2
    """

    def __init__(self, kernel, **kwargs):
        """
        input:
            kernel: pysquid.Kernel object

        """
        self.kernel = kernel        
        n = kernel.N_pad
        m = 2 * n #x and y derivatives
        p = m
        self._oldx = None #warm start for linear solver

        super(Deconvolver, self).__init__(kernel.N_pad, m, p)
        
        #Setup M matrix for Mg = phi
        m = lambda g: self.kernel.applyM(g).real.ravel()
        mt = lambda g: self.kernel.applyMt(g).real.ravel()
        N, N_pad = self.kernel.N, self.kernel.N_pad
        self.M = MyLinearOperator((N, N_pad), matvec = m, rmatvec = mt)

    def f(self, x, phi, **kwargs):
        res = self.M.dot(x) - phi
        return res.dot(res)/2

    def _apply_x_update_kernel(self, x, rho):
        M, A = self.M, self.A
        return M.T.dot(M.dot(x)) + rho * A.T.dot(A.dot(x))

    def _get_x_op(self, rho):
        """
        Create linear operator for x_update
        input:
            rho : float
        """
        N_pad = self.kernel.N_pad
        apply_kernel = lambda x: self._apply_x_update_kernel(x, rho)
        return MyLinearOperator((N_pad, N_pad), 
                                matvec = apply_kernel)

    def start_lagrange_mult(self, x0, z0, rho, phi = None, **kwargs):
        """
        Calculates the initial lagrange multiplier which ensures that
        ADMM is stable if started at the correct x0, z0.

        input:
            x0 : float array of shape self.n, initial x point
            z0 : float array of shape self.m, initial z point
            rho : float, weighting of augmented part of lagrangian
            phi : float array of shape self.
        """
        Op = self._get_x_op(rho)
        A, B, c = self.A, self.B, self.c
        self._y0rhs = -Op.dot(x0) + self.M.T.dot(phi) - rho*A.T.dot(B.dot(z0)-c)
    
        maxiter = kwargs.get('y0_maxiter', None)
        atol = kwargs.get('atol', 1E-6)
        btol = kwargs.get('atol', 1E-6)
        
        self._y0minsol = ssl.lsqr(self.A.T, self._y0rhs, iter_lim = maxiter, 
                                  atol = atol, btol = btol, damp = 1E-5)
        return self._y0minsol[0]

    def x_update(self, z, y, rho, x0 = None, phi = None, **kwargs):
        """
        Perform x_{k+1} = argmin_x L(x, y, z, rho)

        by solving (M^TM + rho A^T A)x = M^Tphi - A^T(y + rho (Bz-c))

        kwargs:
            phi: ndarray of shape self.kernel.N, data to deconvolve
            solver: string of iterative linear solver method. Choose from
                    list linear_solvers
            maxiter: maximum iterations for linear solver
            tol: tolerance for convergence criterion of linear solver

        #TODO: writeup warm start of linear equation solvers
        """
        A, B, c = self.A, self.B, self.c
        self._oldx = self._oldx if self._oldx is not None else np.zeros(self.n)
        maxiter = kwargs.get('maxiter', 200)
        tol = kwargs.get('tol', 1E-6)
        solver_str = kwargs.get('solver', 'minres')

        assert phi is not None, "Must provide phi to deconvolve!"
        assert solver_str in linear_solvers, solver_msg
        solver = getattr(ssl, solver_str)
        
        Op = self._get_x_op(rho)
        self._oldOpx = Op.dot(self._oldx)
        self._rhs = self.M.T.dot(phi) - A.T.dot(y + rho * (B.dot(z) - c))
        self._rhs -= self._oldOpx #warm start

        self._xminsol = solver(Op, self._rhs, maxiter = maxiter, tol = tol)
        self._newx = self._xminsol[0] + self._oldx
        self._oldx = self._newx.copy()
        return self._newx

    def callstart(self, x0 = None, **kwargs):
        self._oldx = None #Reset warm start


class TVDeconvolver(Deconvolver):
    """
    Performs deconvolution of flux image with a
    total variation prior on currents
    """

    def __init__(self, kernel, gamma, **kwargs):
        self.gamma = gamma
        super(TVDeconvolver, self).__init__(kernel, **kwargs)

        Ly_pad, Lx_pad = self.kernel._padshape
        dy, dx = self.kernel.dy, self.kernel.dx
        self.Dh, self.Dv = makeD2_operators((Ly_pad, Lx_pad), dx, dy)

        self.A = vstack([self.Dh, self.Dv])
        self.B = MyLinearOperator((self.m, self.m), matvec = lambda x: -x)
        self.c = np.zeros(self.p)

    def g(self, z):
        dx, dy = z[:self.n], z[self.n:]
        return self.gamma * nu.evaluate('sum(sqrt(dx*dx + dy*dy))')
 
    def _lagrangian_dz(self, z, x, y, rho):
        """
        Evaluate the augmented lagrangian and its derivative
        with respect to z.
        input:
            x, y, z
        returns:
            lagrangian (float), d_lagrangian (m-shaped ndarray)
        """
        r = self.primal_residual(x, z, y)
        gamma = self.gamma
        xx, yy = z[:self.n], z[self.n:]
        tv =  nu.evaluate('sqrt(xx*xx + yy*yy)')
        lagrangian = gamma * tv.sum() + r.dot(y) + rho*r.dot(r)/2
        
        d_tv = np.concatenate((xx/tv, yy/tv))
        d_lagrangian = gamma * d_tv + self.B.T.dot(y + rho*r)
        return lagrangian, d_lagrangian 

    def z_update(self, x, y, rho, z0, **kwargs):
        """
        Evaluate the z-update proximal map for ADMM.
        input:
            z0 : current z-value (2*N_pad)-shaped
            lamb : lagrange multiplier same shape as z0
            Dg : D.dot(g) second derivatives of g
            zsteps : how many LBFGS steps to allow
        output:
            updated z : (2*N_pad)-shaped
        """
        zsteps = kwargs.get('zsteps', 20)
        self._zminsol = minimize(self._lagrangian_dz, z0, args = (x, y, rho,),
                                 method = 'l-bfgs-b', jac = True, 
                                 options = {'maxiter': zsteps})
        return self._zminsol['x']

    def deconvolve(self, phi, x0, g_kwargs = {}, **kwargs):
        f_args = (phi,)
        f_kwargs = {}
        f_kwargs.setdefault('maxiter', kwargs.get('maxiter', 200))
        f_kwargs.setdefault('tol', kwargs.get('tol', 1E-6))
        f_kwargs.setdefault('solver', kwargs.get('solver', 'minres'))

        g_args = ()
        g_kwargs = {}
        g_kwargs.setdefault('zsteps', kwargs.get('zsteps', 40))

        xmin, _, msg = self.minimize(x0, f_args, g_args, f_kwargs, g_kwargs, 
                                     **kwargs)
        return xmin


def test_grad(function, z, h, args):
    z1 = z.copy()
    fd_grad = np.zeros_like(z)
    f0, grad0 = function(z, *args)
    for i in range(len(z)):
        z1[i] += h
        fd_grad[i] = (function(z, *args)[0] - f0)/h
        z1[i] -= h
    return fd_grad, grad0
        



