"""
rnet.py

author: Colin Clement
date: 2015-11-14

This computes currents in a resistor network provided by a mask
in order to approximate current entering and leaving an image.


"""

from __future__ import print_function
import sys
import numpy as np
import scipy as sp
import networkx as nx
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.sparse.linalg import spsolve
from collections import defaultdict
from pysquid.kernels.kernel import BareKernel
from pysquid.kernels.psf import GaussianBlurKernel
from pysquid.component import ModelComponent


class ResistorNetworkModel(ModelComponent):
    """
    An object for modeling arbitrary linear surface currents

    """
    def __init__(self, mask, kernel = None, 
                 phi_offset = None, gshape = None,
                 padding = None, **kwargs):
        """
        Initialize a resistor network. 
        mask : Boolean or int array of shape (Ly, Lx) for determining
                which elements of mesh array should have loop currents
        kernel : instance of Kernel class for computing magnetic flux

        phi_offset : position of g-field corner in mask for aligning
                    the mask and the sample data

        gshape : (Ly, Lx) tuple of ints, shape of sample g-field
        
        padding : (py, px) tuple of ints, padding of sample g for model

        kwargs: 
        deltaV :    (float) is the applied voltage 
                    (to the top and bottom unless otherwise specified)
        electrodes : list of two integers [cathode, anode]. Default is 
                     [0, Lx], i.e. the top two corners
                     Specify in original mask units.
        
        """
        self.mask = mask
        self.flatmask = self.mask.flatten()
        super(ResistorNetworkModel, self).__init__(mask.shape, **kwargs)
        
        self.kernel = kernel
        self.phi_offset = phi_offset if phi_offset is not None else [0, 0]
        self.gshape = gshape if gshape is not None else (self.Ly, self.Lx)
        self.padding = padding if padding is not None else [0, 0]
        self._blur = kwargs.get('blur', [2., 2.])
        self.blurkernel = GaussianBlurKernel(self._shape, self._blur, 
                                             dx = 1., dy = 1.)

        self.deltaV = kwargs.get('deltaV', 1.)
        self.J_ext = np.array([1.])

        self._setupGridsNumbers()

        if 'electrodes' in kwargs:
            self.electrodes = np.array(kwargs['electrodes'])
            condition = (self.electrodes < 0) + (self.electrodes > self.Lx)
            if np.any(condition):
                raise RuntimeError("Please select electrodes on the top edge of"+
                                   " the mask, inside [{}, {}]".format(0, self.Lx))
        else:
            self.electrodes = np.array([0, self.Lx])

        self._constructSparseMatricesAndCholesky()
        self.solveRnet()


    def __setstate__(self, d):
        self.__dict__ = d
        self._setupGridsNumbers()

    #==============================
    #   square grid of resistors
    #==============================
    
    def _setupGridsNumbers(self):
        Ly, Lx = self.Ly, self.Lx
        self.N_nodes = (Lx+1)*(Ly+1)
        self.N_hor = self.N + Lx
        self.N_ver = self.N + Ly
        self.evaluations = 0
        self.N_singular_evals = 0
        self._globalLoopField = np.zeros((Ly, Lx))
        
        maskx = np.arange( 1/2.,  Lx,  1)
        masky = np.arange(-1/2., -Ly, -1)
        self.maskx_g, self.masky_g = np.meshgrid(maskx, masky)
        points = np.c_[self.maskx_g.ravel(), self.masky_g.ravel()]
        x = np.arange( 0.5,  Lx)
        y = np.arange(-0.5, -Ly, -1)
        self.x_g, self.y_g = np.meshgrid(x, y)
      
        self.maskInterpolator = NearestNDInterpolator(points, self.mask.ravel())
        self.gmask = self.maskInterpolator(np.c_[self.x_g.ravel(),
                                                 self.y_g.ravel()])
        self._unPickleable = ['maskInterpolator']
        self.gmaskflat = self.gmask.ravel()

        try:    #Only make if don't exist
            self.gfield
        except AttributeError:
            self.gfield = np.zeros(self.N)
            self.gfieldflat = self.gfield.flatten()
            self.gfieldpadded = np.zeros((2*Ly, 2*Lx))
        
        #Mesh coordinates
        self.meshIndex = (np.cumsum(self.gmask.ravel())-1).astype('int')
        self.N_meshes = max(self.meshIndex)+1
        self.N_loops = self.N_meshes + 1
        self.voltageConst = np.zeros(self.N_loops)
        self.voltageConst[-1] = self.deltaV

    def _constructMeshMatrices(self):
        """
        This function computes self.meshMatrix as a sparse matrix
        by loops over the meshes of a square grid. It also computes
        a dictionary of meshes whose values are the edges contained
        and a dictionary of edges whose values are the meshes which
        contain them.

        """   
        row, col, data = [], [], []
        neigh = np.array([[-1,0], [0,1], [1,0], [0,-1]]) #left, down, right, up
        Lx, Ly, N_hor = self.Lx, self.Ly, self.N_hor
        self.meshEdges = defaultdict(list) # {'mesh': [list of edges]}
        self.edgeMeshes = defaultdict(list)# {'edge': [list of meshes]}
        self.G = nx.Graph() 
        
        for i in np.arange(self.N)[self.gmask > 0]:
            mesh = self.meshIndex[i] #meshes don't cound empty spots
            mesh_coord = np.array([i % Lx, i//Lx])
            mesh_edges = [i+i//Lx+N_hor, i+Lx, i+i//Lx+N_hor+1, i] #CCW order
            self.meshEdges[mesh] = mesh_edges
            row += [mesh]
            col += [mesh]
            data += [4.] #change later
            
            upper_left = i+i//Lx
            self.G.add_edge(upper_left, upper_left+1, label=i)
            self.G.add_edge(upper_left, upper_left+Lx+1, label=i+N_hor+i//Lx)
            
            for edge, n in zip(mesh_edges, neigh):
                self.edgeMeshes[edge] += [mesh]
                neigh_x, neigh_y = mesh_coord + n
                neigh_mesh =  neigh_x + neigh_y*Lx
                if 0<=neigh_x<Lx and 0<=neigh_y<Ly:
                    inside = True
                    mesh_neigh_no = self.meshIndex[neigh_mesh]
                    neigh_mask = self.gmaskflat[neigh_mesh]
                    if neigh_mask:
                        row += [mesh]
                        col += [mesh_neigh_no]
                        data += [-1.] #change later
                else:
                    inside = False
                    neigh_mask = 0

                if not inside or not neigh_mask:
                    if np.all(n == [1,0]):
                        start = upper_left + 1
                        lab = i+N_hor+1+i//Lx
                        end = upper_left + 1 + Lx + 1
                        self.G.add_edge(start, end, label = lab)
                    elif np.all(n == [0,1]):
                        start = upper_left + Lx + 1
                        lab = i+Lx
                        end = upper_left + 1 + Lx + 1
                        self.G.add_edge(start, end, label = lab)
        self.N_currents = len(self.G.edges()) #depends on mask
        return self._appliedVoltageGeometry(self.G, row, col, data)

    def _appliedVoltageGeometry(self, nxGraph, row, col, data):
        """
        Given nxGraph, row, col, and data from self._constructMeshMatrices,
        adds to row, col, and data a global voltage loop between the cathode
        and anode nodes in the self.electrodes list. Uses A* to compute the
        shortest path between them, returns (data, (row, col)) in the format
        that scipy.sparse.coo_matrix uses.
        w
        """
        G = nxGraph
        cathode, anode = self.electrodes
        nPath = nx.astar_path(G, cathode, anode)
        self._voltagePath = np.array([G.edge[nPath[n]][nPath[n+1]]['label'] 
                                      for n in range(len(nPath)-1)])
        self.meshEdges[self.N_loops-1] = self._voltagePath.copy()
        voltageOrient = []
        for edge, n1, n2 in zip(self._voltagePath, nPath[:-1], nPath[1:]):
            voltageOrient += [1 if n1 < n2 else -1]
            for m in self.edgeMeshes[edge]:
                #NOTE: Assumes meshEdges were made in CCW order as above
                try:
                    loopOrient = [1,1,-1,-1][self.meshEdges[m].index(edge)]
                except ValueError:
                    print(edge, m)
                    raise

                row += [m]
                col += [self.N_loops-1]
                data += [voltageOrient[-1] * loopOrient]

                row += [self.N_loops-1]
                col += [m]
                data += [voltageOrient[-1] * loopOrient]
            self.edgeMeshes[edge] += [self.N_loops-1]

        row += [self.N_loops-1]
        col += [self.N_loops-1]
        data += [len(self._voltagePath)]
    
        self._voltageOrient = np.array(voltageOrient)
        
        #Find squares inside global loop
        topclosed = np.arange(*self.electrodes)
        self._voltagePath = np.r_[self._voltagePath, topclosed]
        self._voltageOrient = np.r_[self._voltageOrient, 
                                    -1*np.ones_like(topclosed)]
        hpaths = np.zeros((self.Ly+1, self.Lx))
        vpaths = np.zeros((self.Ly, self.Lx+1))
        h_indices = self._voltagePath[self._voltagePath < self.N_hor]
        v_indices = self._voltagePath[self._voltagePath > self.N_hor]-self.N_hor
        for hh in h_indices:
            hpaths[hh/self.Lx, hh%self.Lx] = 1.
        for vv in v_indices:
            vpaths[vv/(self.Lx+1), vv % (self.Lx+1)] = 1.
        hfill = np.cumsum(hpaths, axis=0)[:-1] == 1
        vfill = np.cumsum(vpaths, axis=1)[:,:-1] == 1
        self._globalLoopPatch = 1 * (hfill * vfill)
        return (data, (row, col))

    #==============================
    #   Sparse Matrices/Cholesky Stuff 
    #==============================
    
    def _constructSparseMatricesAndCholesky(self):
        N_loops = self.N_loops
        spRmatrix = self._constructMeshMatrices()
        self.meshRMatrix = sp.sparse.coo_matrix(spRmatrix, 
                                                (N_loops, N_loops)).tocsc()

    #==============================
    #   Calculate currents and fields
    #==============================

    def solveRnet(self):
        self.meshCurrents = spsolve(self.meshRMatrix, self.voltageConst)
        self.gfieldflat[self.gmaskflat == 1] = self.meshCurrents[:-1].copy()
        self.gfield = self.gfieldflat.reshape(self.Ly, self.Lx)
        self.gfield += self.meshCurrents[-1] * self._globalLoopPatch
        if np.any(np.isnan(self.meshCurrents)):
            self.N_singular_evals += 1
            return True
        self.gfield[:,:] = self.blurkernel.applyM(self.gfield).real
        return False

    @property
    def _cutp(self):
        oy, ox = self.phi_offset
        Ly, Lx = self.gshape #This is NOT self.Lx/self.Ly!
        py, px = self.padding
        return np.s_[oy-py:oy+Ly+py, ox-px:ox+Lx+px]
    
    @property
    def _cut(self):
        oy, ox = self.phi_offset
        Ly, Lx = self.gshape #This is NOT self.Lx/self.Ly!
        return np.s_[oy:oy+Ly, ox:ox+Lx]

    @property
    def _nop(self):
        Ly, Lx = self.gshape #This is NOT self.Lx/self.Ly!
        py, px = self.padding
        return np.s_[py:py+Ly,px:px+Lx]

    def updateParams(self, name, values):
        """
        This object formally relies only upon J_ext,
        all psf_params dependence is handled  by the kernel

        """
        if name == 'J_ext':
            self.J_ext = values[0] 
        self._updateModel_g()
        self._updateField()
        self.ext_flux = self.J_ext * self._unitJ_flux[self._cut]

    def _updateField(self):
        if self.kernel:
            self._unitJ_flux = self.kernel.applyM(self.gfield).real
        else:
            self._unitJ_flux = np.zeros_like(self.mask)
            print('No kernel attribute\n')

    def _updateModel_g(self):
        """ 
        Compute gfield and its curvatures for fitting
        """
        J = self.J_ext
        cutp = self._cutp
        nop, cut = self._nop, self._cut
        self.ext_g = J * self.gfield[cutp]

    def computeGradients(self):
        if self.kernel:
            self._d_unitJ_flux = self.kernel.computeGradients(self.gfield)
            self.d_ext_flux = self.J_ext*self._d_unitJ_flux[self._cut]  
        else:
            self.d_ext_flux = np.zeros_like(self.ext_flux)
        return self.d_ext_flux

    @property
    def currents(self):
        return curl(self.gfield)



