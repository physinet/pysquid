"""
admm.py

author: Colin Clement
date: 2016-12-06

This defines an abstract class for implementing
the Alternating Difference Method of Multipliers
based on Stephen Boyd's 2010 paper

    http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

This class must be subclassed to be useful. A proper subclass
must define the methods ADMM.f, ADMM.g, ADMM.x_update, ADMM.z_update,
and it must define the attributes A, B, and c, defined below.
"""

import numpy as np


class ADMM(object):
    """
    Solves the optimization problem

    x* = argmin_x f(x) + g(z)
        subject to Ax + Bz = c

    where x is in R^n, y is in R^m, and C is in R^p.
    Matrices A and B are shaped appropriately and f and
    g are convex.

    The algorithm is as follows:
        define the augmented lagrangian 
        L(x, z, y, p) := f(x) + g(z) + y.dot(Ax+Bz-c)
                         + rho/2 ||Ax+bz-c||^2
        repeat until convergence criterion met:
            x_{k+1} := argmin_x L(x_{k}, z_{k}, y_{k}, p)
            z_{k+1} := argmin_z L(x_{k+1}, z, y_{k}, p)
            y_{k+1} := y_{k} + rho(Ax_{k+1} + Bz_{k+1} - c)

    The notation and most of the algorithm are implementations
    of the paper "Distributed Optimization and Statistical
    Learning via the Alternating Direction Method of Multipliers"
    by Stephen Boyd et al. Foundations and Trends in Machine 
    Learning Vol. 3, No. 1 (2010)
    http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    """

    def __init__(self, n, m, p, **kwargs):
        """
        n : (int) size of x
        m : (int) size of z
        p : (int) size of c

        kwargs:
            rho : (float) augmented regularization
        """
        self.n = n
        self.m = m
        self.p = p

        self.A = None
        self.B = None
        self.c = None

        self.msg = {1: "Convergence criterion satisfied",
                    2: "Iteration limit reached"}

    def f(self, x, *args, **kwargs):
        """
        One of the functions to be minimized.
        input:
            x: numpy array of shape n
        """
        pass

    def g(self, z, *args, **kwargs):
        """
        Second functions to be minimized.
        input:
            z: numpy array of shape m
        """
        pass

    def x_update(self, z, y, rho, x0 = None, *args, **kwargs):
        """
        Perform the optimization 
            x_{k+1} := argmin_x L(x_{k}, z_{k}, y_{k}, p)
        """
        pass

    def z_update(self, x, y, rho, z0 = None, *args, **kwargs):
        """
        Perform the optimization 
            z_{k+1} := argmin_z L(x_{k+1}, z, y_{k}, p)
        """
        pass

    def callstart(self, x0 = None, *args, **kwargs):
        """
        Call at the beginning of minize for custom child classes
        """
        pass

    def callback(self, *args, **kwargs):
        """
        Callback function for each iteration
        """
        pass

    def y_update(self, y, x, z, rho):
        """
        Update lagrange multiplier
        """ 
        A, B, c = self.A, self.B, self.c
        return y + rho * (A.dot(x) + B.dot(z) - c)

    def cost(self, x, z, f_args = (), g_args = ()):
        """
        Compute objective function
        """
        return self.f(x, *f_args) + self.g(z, *g_args)

    def check_shape(self, arr, name, shape):
        errstr = "{} should have shape {}".format(name, shape)
        assert arr.shape == shape, errstr

    def primal_residual(self, x, z, y):
        """
        Compute the primal residual, the violation
        of the condition Ax+Bz=c
        """
        A, B, c = self.A, self.B, self.c
        return A.dot(x) + B.dot(z) - c

    def dual_residual(self, z1, z0, rho):
        """
        Compute the dual residual, another quantity
        guaranteed to converge in ADMM
        """
        A, B, c = self.A, self.B, self.c
        return rho * A.T.dot(B.dot(z1 - z0))

    def stop(self, r, s, x, z, y, eps_rel, eps_abs, iprint = 0):
        """
        Compute the stopping criterion of ADMM as discussed
        in section 3.3.1 of Boyd 2010.
        """
        A, B, c = self.A, self.B, self.c
        Ax = self.A.dot(x)
        Bz = self.B.dot(z)
        ATy = self.A.T.dot(y)
        eps_pri = (np.sqrt(self.p) * eps_abs
                   + eps_rel * max(Ax.dot(Ax), Bz.dot(Bz), c.dot(c)))
        eps_dual = (np.sqrt(self.n) * eps_abs 
                    + eps_rel * ATy.dot(ATy))
        if r.dot(r) <= eps_pri and s.dot(s) <= eps_dual:
            return True
        else:
            return False

    def update_rho(self, rho, t_inc, t_dec, mu, r, s):
        """
        See equation 3.13, this update tries to keep
        the dual and primal residuals within a factor of
        mu of each other
        """
        r2 = r.dot(r)
        s2 = s.dot(s)
        if r2 > mu*s2:
            return rho * t_inc
        elif s2 > mu*r2:
            return rho /t_dec
        else:
            return rho

    def minimize(self, x0 = None, f_args = (), g_args = (),
                 f_kwargs = {}, g_kwargs = {}, **kwargs):
        """
        Implementation of the ADMM algorithm as described
        in Boyd 2010. Includes the scheme for updating
        rho as described in section 3.4.1
        input:
            x0: ndarray of length n, initial x. Defaults 
                to random array
        returns:
            optimal x0, message integer

        kwargs:

        z0 : ndarray of length m, initial z. Defaults to
            random array
        y0 : ndarray of length p, initial y. Defaults to 
            random array
        rho : float, initial weighting of augmented term

        Parameters effecting convergence of algorithm

        eps_abs : float, tolerance for absolute size of residuals
        eps_rel : float, tol for relative size of residuals

        See equation 3.13 in Boyd 2010 for definition of these.
        t_inc : Amount to increase rho by when primal residual is large
        t_dec : Amount to decrease rho by when dual residual is large
        mu : Algorithm tries to keep ||dual||^2 = mu * ||primal||^2
        """
        self.callstart(x0, **kwargs)

        x0 = x0 if x0 is not None else np.random.randn(self.n)
        z0 = kwargs.get("z0", np.random.randn(self.m))
        y0 = kwargs.get("y0", np.random.randn(self.p))
        iprint = kwargs.get('iprint', 1)
    
        self.check_shape(x0, "x0", (self.n,))
        self.check_shape(z0, "z0", (self.m,))
        self.check_shape(y0, "y0", (self.p,))

        itnlim = kwargs.get("itnlim", 20)
        eps_abs = kwargs.get("eps_abs", 1E-2)
        eps_rel = kwargs.get("eps_rel", 1E-4)
        t_inc = kwargs.get("t_inc", 2)
        t_dec = kwargs.get("t_dec", 2)
        mu = kwargs.get("mu", 10)
        rho = kwargs.get("rho", 1E-2)

        #Check types
        assert isinstance(itnlim, int), "itnlim must be an int"
        assert t_inc > 0, "t_inc must > 0"
        assert t_dec > 0, "t_dec must be 0"
        assert mu > 0, "mu must be 0"
        assert rho > 0, "rho must be 0"

        r0 = self.primal_residual(x0, z0, y0)
        c0 = self.cost(x0, z0, f_args, g_args)

        if iprint:
            print("Initial cost = {:e}".format(c0))

        for itn in range(itnlim):
            x1 = self.x_update(z0, y0, rho, x0, *f_args, **f_kwargs)
            z1 = self.z_update(x1, y0, rho, z0, *g_args, **g_kwargs)
            y1 = self.y_update(y0, x1, z1, rho)

            r1 = self.primal_residual(x1, z1, y1)
            s = self.dual_residual(z1, z0, rho)

            c1 = self.cost(x1, z1, f_args, g_args)
            if self.stop(r1, s, x1, z1, y1, eps_rel, eps_abs, iprint):
                msg = 1
                break
            else:
                if iprint > 1:
                    pstr = ("\tItn {:1}: cost = {:e}, r = {:.3e}, s = {:.3e} " + 
                            "rho = {}")
                    print(pstr.format(itn, c1, r1.dot(r1), s.dot(s), rho))

            rho = self.update_rho(rho, t_inc, t_dec, mu, r1, s)

            self.callback(x1, z1, y1)
        
        if itn == itnlim - 1:
            msg = 2

        if iprint:
            print(self.msg[msg])

        return x1, z1, msg
        

