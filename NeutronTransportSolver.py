from mpi4py import MPI
from dolfinx import *
from dolfinx import mesh, fem, default_scalar_type
import numpy as np
from dolfinx.fem.petsc import assemble_matrix
from mshr import *
import basix, ufl, os
import scipy as sp
from PowerIteration import gen_power_it_dynamic_momentum

class NeutronTransportSolver:
    '''
    Solver two group for the Neutron transport equation at steady state.
    This solver calculates the multiplication factor and the 2D or 3D fluxes for the thermal and fast group.
    Input:
    - Domain: mesh
    - Equation constants: D1, D2, Sa1, Sa2, nusigf1, nusigf2, S12
    - k: orden of polinomials used for the afinite element aproximation

    '''
    def __init__(
            self,
            domain,
            D1 = 1.0, D2 = 0.5,
            Sa1 = 0.2, Sa2 = 0.1,
            nusigf1 = 0.3, nusigf2 = 0.1,
            S12 = 0.1,
            k = 1,
            bord_cond = 'dir'       
    ):
        self.domain = domain
        self.D1, self.D2 = D1, D2
        self.Sa1, self.Sa2 = Sa1, Sa2
        self.nusigf1, self.nusigf2 = nusigf1, nusigf2
        self.S12 = S12
        self.k = k
        self.bord_cond = bord_cond

        self.V = self._function_space()
        self.eigval = None
        self.vr = None
        self.vi = None
        self.phi1 = None
        self.phi2 = None
        self.phi1_list = None
        self.phi2_list = None
        self.a = None
        self.f = None

    def _function_space(self):
        H = basix.ufl.element("Lagrange", self.domain.basix_cell(), self.k)
        Vm = basix.ufl.mixed_element([H,H])
        V = fem.functionspace(self.domain, Vm)
        return V
    
    def solve(self):
        phi1, phi2 = ufl.TrialFunctions(self.V)
        v1, v2 = ufl.TestFunctions(self.V)

        dx = ufl.dx

        #Formas bilineales A y F
        A = self.D1 * ufl.inner(ufl.grad(phi1), ufl.grad(v1)) *dx
        A+= (self.Sa1 + self.S12) * phi1 * v1 *dx
        A+= self.D2 * ufl.inner(ufl.grad(phi2), ufl.grad(v2)) *dx
        A+= self.Sa2 * phi2 * v2*dx
        A-= self.S12 * phi1 * v2 * dx

        F = (self.nusigf1 * phi1 * v1 + self.nusigf2 * phi2 * v1) * dx

        def boundary_all(x):
            return np.full(x.shape[1], True, dtype=bool)

        if self.bord_cond == 'dir':  
            boundary_facets = mesh.locate_entities_boundary(self.domain, 
                                                            self.domain.topology.dim - 1, boundary_all)
            boundary_dofs_x = fem.locate_dofs_topological(self.V.sub(0), 
                                                          self.domain.topology.dim - 1, boundary_facets)
            boundary_dofs_x2 = fem.locate_dofs_topological(self.V.sub(1), 
                                                           self.domain.topology.dim - 1, boundary_facets)

            bcx = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, self.V.sub(0))
            bc1x = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x2, self.V.sub(1))
            bcs = [bc1x, bcx]
        elif self.bord_cond == 'neu':
            bcs = []
        else:
            raise ValueError("Condición de borde inexistente: debe ser 'dir' o 'neu'")

        #ensamble del sistema
        a = assemble_matrix(fem.form(A), bcs = bcs, diagonal = 1e2)
        a.assemble()
        f = assemble_matrix(fem.form(F), bcs = bcs, diagonal = 1e-2)
        f.assemble()
        self.a = a
        self.f = f

        v0 =self.a.createVecLeft()
        v0.set(1.0)

        lam_k, x_k, k, res = gen_power_it_dynamic_momentum(self.f, self.a, v0, tol = 1e-8, max_iter = 1000)
        self.eigval = lam_k
        self.power_its = k
        self.vec = x_k
        self.power_res = res

        phi = fem.Function(self.V)
        phi.x.array[:] = self.vec.array
        phi1, phi2 = phi.split()
        V0 = fem.functionspace(self.domain, ('CG', 1))

        self.phi1_proj = fem.Function(V0)
        self.phi1_proj.interpolate(fem.Expression(phi1, V0.element.interpolation_points()))

        self.phi2_proj = fem.Function(V0)
        self.phi2_proj.interpolate(fem.Expression(phi2, V0.element.interpolation_points()))

    def phi_norms(self, num = 0):
        phi1_norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.phi1_proj) * ufl.dx)))
        phi2_norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.phi2_proj) * ufl.dx)))

        return phi1_norm, phi2_norm
    
    def export(self, name = 'result'):
        
        path = f"outputs/{name}"
        if MPI.COMM_WORLD.rank == 0 and not os.path.exists(path):
            os.makedirs(path)

        with io.VTKFile(MPI.COMM_WORLD, f"{path}/phi1_proj.pvd", "w") as vtk:
            vtk.write_function(self.phi1_proj)

        with io.VTKFile(MPI.COMM_WORLD, f"{path}/phi2_proj.pvd", "w") as vtk:
            vtk.write_function(self.phi2_proj)

        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.rank == 0:
            print("Archivos guardados en: ", path)
