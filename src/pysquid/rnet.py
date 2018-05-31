"""
rnet.py

author: Colin Clement
date: 2015-11-14

This computes currents in a resistor network provided by a mask
in order to approximate current entering and leaving an image.


"""

from __future__ import print_function

from pysquid.kernels.kernel import BareKernel
from pysquid.kernels.psf import GaussianBlurKernel
from pysquid.component import ModelComponent
from pysquid.util.graph import Graph
from pysquid.util.helpers import curl

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from collections import defaultdict


class ResistorNetworkModel(ModelComponent):
    """
    An object for modeling arbitrary linear surface currents

    """
    def __init__(self, mask, kernel=None, phi_offset=None, gshape=None,
                 padding=None, **kwargs):
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
        resistors : array_like of shape ((Lx+1)*(Ly+1)) 
            resistances of mesh surrounding mask pixels
        deltaV :    (float) is the applied voltage
                    (to the top and bottom unless otherwise specified)
        electrodes : list of two integers [cathode, anode]. Default is
                     [0, Lx], i.e. the top two corners
                     Specify in original mask units.
        """
        self.mask = mask
        super(ResistorNetworkModel, self).__init__(mask.shape, **kwargs)

        self.deltaV = kwargs.get('deltaV', 1.)
        self._setupGridsNumbers()
        Ly, Lx = self.Ly, self.Lx

        self.kernel = kernel
        self.phi_offset = phi_offset if phi_offset is not None else [0, 0]
        self.gshape = gshape if gshape is not None else (Ly, Lx)
        self.padding = padding if padding is not None else [0, 0]
        self.resistors = kwargs.get('resistors', np.ones(self.N_hor+self.N_ver))

        self.electrodes = kwargs.get('electrodes', np.array([0, self.Lx]))

        row, col, data, G = self._constructMeshMatrices()
        sparsemat = self._appliedVoltageGeometry(row, col, data, G,
                                                 *self.electrodes)
        self._constructSparseMatrix(sparsemat)
        self.solveRnet()
        self.updateParams('J_ext', [1.])

    def __setstate__(self, d):
        self.__dict__ = d
        self._setupGridsNumbers()

    # ==============================
    #   square grid of resistors
    # ==============================

    def _setupGridsNumbers(self):
        #  Re-evaluate whether we need this complex interpolation stuff in here
        Ly, Lx = self.Ly, self.Lx
        self.N_nodes = (Lx+1)*(Ly+1)
        self.N_hor = self.N + Lx
        self.N_ver = self.N + Ly
        self._globalLoopField = np.zeros((Ly, Lx))

        self.maskflat = self.mask.flatten()

        # Only make if don't exist
        try:
            self.gfield
        except AttributeError:
            self.gfield = np.zeros(self.N)
            self.gfieldflat = self.gfield.flatten()
            self.gfieldpadded = np.zeros((2*Ly, 2*Lx))

        # Mesh coordinates
        self.meshIndex = (np.cumsum(self.maskflat)-1).astype('int')
        self.N_meshes = max(self.meshIndex)+1
        self.N_loops = self.N_meshes + 1
        self.voltageConst = np.zeros(self.N_loops)
        self.voltageConst[-1] = self.deltaV

    def meshEdgeIndices(self, i):
        # left, bottom, right, top
        return [i + i//self.Lx + self.N_hor, i + self.Lx, 
                i + i//self.Lx + self.N_hor+1, i]  # CCW

    def meshIndexCornerIndices(self, i):
        ul = i + i // self.Lx
        # upper-left, lower-left, lower-right, upper-right
        return [ul, ul + self.Lx + 1, ul + self.Lx + 2, ul + 1]

    def _constructMeshMatrices(self):
        """
        This function computes self.meshMatrix as a sparse matrix
        by loops over the meshes of a square grid. It also computes
        a dictionary of meshes whose values are the edges contained
        and a dictionary of edges whose values are the meshes which
        contain them.
        """
        row, col, data = [], [], []
        neigh = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # leftdownrightup
        Lx, Ly, N_hor = self.Lx, self.Ly, self.N_hor
        self.meshEdges = defaultdict(list)   # {'mesh': [list of edges]}
        self.edgeMeshes = defaultdict(list)  # {'edge': [list of meshes]}
        G = Graph((Lx+1) * (Ly+1))

        for i in np.arange(self.N)[self.maskflat > 0]:
            mesh = self.meshIndex[i]  # meshes don't cound empty spots
            mesh_coord = np.array([i % Lx, i//Lx])  # x, y coordinates
            mesh_edges = self.meshEdgeIndices(i)

            self.meshEdges[mesh] = mesh_edges
            row += [mesh]
            col += [mesh]
            data += [self.resistors[mesh_edges].sum()]

            ul, ll, lr, ur = self.meshIndexCornerIndices(i)
            G.insert(ul, ur, i)  # w = resistor label
            G.insert(ul, ll, i + N_hor + i//Lx)

            for edge, n in zip(mesh_edges, neigh):
                self.edgeMeshes[edge] += [mesh]
                neigh_x, neigh_y = mesh_coord + n
                neigh_mesh = neigh_x + neigh_y*Lx
                if 0 <= neigh_x < Lx and 0 <= neigh_y < Ly:
                    inside = True
                    mesh_neigh_no = self.meshIndex[neigh_mesh]
                    neigh_mask = self.maskflat[neigh_mesh]
                    if neigh_mask:
                        row += [mesh]
                        col += [mesh_neigh_no]
                        data += [-1 * self.resistors[edge]]  # counter rotating
                else:
                    inside = False
                    neigh_mask = 0

                if not inside or not neigh_mask:
                    if np.all(n == [1, 0]):
                        G.insert(ur, lr, ul + 1 + Lx + 1)
                    elif np.all(n == [0, 1]):
                        G.insert(ll, lr, i + Lx)
        self.N_currents = G.ne
        return row, col, data, G

    def _appliedVoltageGeometry(self, row, col, data, G, cathode, anode):
        """
        Given nxGraph, row, col, and data from self._constructMeshMatrices,
        adds to row, col, and data a global voltage loop between the cathode
        and anode nodes in the self.electrodes list. Uses A* to compute the
        shortest path between them, returns (data, (row, col)) in the format
        that scipy.sparse.coo_matrix uses.
        """
        # Find shortest path between cathode and anode
        path = G.findpath(cathode, anode, G.bfs(cathode))

        # Reconstruct edge path
        self._global_edge_path = []
        for x, y in zip(path[:-1], path[1:]):
            v = G.edges[x]
            while v:
                if v.y == y:
                    self._global_edge_path.append(v.w)
                    break
                v = v.nextedge
        self._global_edge_path = np.array(self._global_edge_path)

        self.meshEdges[self.N_loops-1] = self._global_edge_path
        orientation = [1, 1, -1, -1]  # leftdownrightup order as above
        for edge, n1, n2 in zip(self._global_edge_path, path[:-1], path[1:]):
            for m in self.edgeMeshes[edge]:
                orient_edge = orientation[self.meshEdges[m].index(edge)]
                orient_edge = (2*(n1 < n2)-1)*orient_edge

                row += [m]
                col += [self.N_loops-1]
                data += [self.resistors[edge]*orient_edge]

                row += [self.N_loops-1]
                col += [m]
                data += [self.resistors[edge]*orient_edge]
            
            self.edgeMeshes[edge] += [self.N_loops-1]

        row += [self.N_loops-1]
        col += [self.N_loops-1]
        data += [self.resistors[self._global_edge_path].sum()]

        self._globalLoopPatch = self._appliedLoopGPatch(self._global_edge_path)
        return (data, (row, col))

    def _appliedLoopGPatch(self, edge_path):
        """ 
        Convention: right-hand-rule, CCW currents are positive so that B-field
        is positive out of the page
        """
        hor = np.zeros((self.Ly + 1, self.Lx))
        ver = np.zeros((self.Ly, self.Lx + 1))
        for e in edge_path:
            if e < self.N_hor:  # horizontal currents
                hor[e//self.Lx, e % self.Lx] = 1.
            else:  # vertical currents
                ver[(e - self.N_hor)//(self.Lx + 1), 
                    (e - self.N_hor) % (self.Lx + 1)] = 1.
        return np.cumsum(hor, 1)[:-1] * np.cumsum(ver, 0)[:,:-1]

    def _constructSparseMatrix(self, sprmat):
        N = self.N_loops
        self.meshRMatrix = coo_matrix(sprmat, (N, N)).tocsc()

    # ==============================
    #   Calculate currents and fields
    # ==============================

    def solveRnet(self):
        self.meshCurrents = spsolve(self.meshRMatrix, self.voltageConst)
        self.gfieldflat[self.maskflat == 1] = self.meshCurrents[:-1].copy()
        self.gfield = self.gfieldflat.reshape(self.Ly, self.Lx)
        self.gfield += self.meshCurrents[-1] * self._globalLoopPatch
        return False

    @property
    def _cutp(self):
        oy, ox = self.phi_offset
        Ly, Lx = self.gshape  # This is NOT self.Lx/self.Ly!
        py, px = self.padding
        return np.s_[oy-py:oy+Ly+py, ox-px:ox+Lx+px]

    @property
    def _cut(self):
        oy, ox = self.phi_offset
        Ly, Lx = self.gshape  # This is NOT self.Lx/self.Ly!
        return np.s_[oy:oy+Ly, ox:ox+Lx]

    @property
    def _nop(self):
        Ly, Lx = self.gshape  # This is NOT self.Lx/self.Ly!
        py, px = self.padding
        return np.s_[py:py+Ly, px:px+Lx]

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
