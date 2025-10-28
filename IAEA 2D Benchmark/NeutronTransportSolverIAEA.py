from mpi4py import MPI
from dolfinx import *
from dolfinx import fem, default_scalar_type
import numpy as np
from slepc4py import SLEPc
from mshr import *
import ufl
import os
import basix
import dolfinx.fem.petsc as fem_petsc
from PowerIteration import gen_power_it_dynamic_momentum



class NeutronTransportSolver2:
    def __init__(
            self,
            domain,
            
            D1M=2.0, D2M=0.3,
            Sa1M=1e-8, Sa2M=0.01,
            nusigf1M=1e-8, nusigf2M=1e-8,
            S12M=0.04,

            D1F=5.0, D2F=0.4,
            Sa1F=0.01, Sa2F=0.08,
            nusigf1F=0.135, nusigf2F=0.135,
            S12F=0.02,

            D13=10.0, D23=0.4,
            Sa13=0.01, Sa23=0.085,
            nusigf13=0.135, nusigf23=0.135,
            S123=0.02,

            D14=15.0, D24=0.4,
            Sa14=0.01, Sa24=0.13,
            nusigf14=0.135, nusigf24=0.135,
            S124=0.02,
            N_eig=4,
            k=1,
            bord_cond='dir',
            cell_tags=None,
            facet_tags=None,
            ids=None,
            power_tol = 1e-5,
            ksp_tol = 1e-5,
            eigmethod = 'powerit'
    ):
        self.domain = domain
        self.cell_tags = cell_tags
        self.eigmethod = eigmethod
        self.D1M, self.D2M = D1M, D2M
        self.Sa1M, self.Sa2M = Sa1M, Sa2M
        self.nusigf1M, self.nusigf2M = nusigf1M, nusigf2M
        self.S12M = S12M
        self.D1F, self.D2F = D1F, D2F
        self.Sa1F, self.Sa2F = Sa1F, Sa2F
        self.nusigf1F, self.nusigf2F = nusigf1F, nusigf2F
        self.S12F = S12F

        self.D13, self.D23 = D13, D23
        self.Sa13, self.Sa23 = Sa13, Sa23
        self.nusigf13, self.nusigf23 = nusigf13, nusigf23
        self.S123 = S123
        self.D14, self.D24 = D14, D24
        self.Sa14, self.Sa24 = Sa14, Sa24
        self.nusigf14, self.nusigf24 = nusigf14, nusigf24
        self.S124 = S124

        self.N_eig = N_eig
        self.k = k
        self.bord_cond = bord_cond
        self.facet_tags = facet_tags
        self.V = self._function_space()
        self.eigvals = None
        self.vr = None
        self.vi = None
        self.phi1 = None
        self.phi2 = None
        self.phi1_list = None
        self.phi2_list = None
        self.power_tol = power_tol
        self.ksp_tol = ksp_tol
        if ids is None:
            self.MOD, self.FUEL, self.G_HEX, self.G_IFACE = 1, 2, 11, 12
        else:
            self.MOD, self.FUEL, self.G_HEX, self.G_IFACE = ids


    def _function_space(self):
        H = basix.ufl.element("Lagrange", self.domain.basix_cell(), self.k)
        Vm = basix.ufl.mixed_element([H, H])
        V = fem.functionspace(self.domain, Vm)
        return V


    def solve(self):
        phi1, phi2 = ufl.TrialFunctions(self.V)
        v1, v2 = ufl.TestFunctions(self.V)
        dx = ufl.Measure("dx", domain=self.domain,
                         subdomain_data=self.cell_tags)
        ds = ufl.Measure("ds", domain=self.domain,
                         subdomain_data=self.facet_tags)
                         
        available_tags = np.unique(self.cell_tags.values)

        if 1 in available_tags:
            A = self.D1M*ufl.inner(ufl.grad(phi1), ufl.grad(v1))*dx(1)
            A += self.D1M*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(1)
            A += (self.Sa1M + self.S12M)*phi1*v1*dx(1)
            A += self.D2M*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(1)
            A += self.Sa2M*phi2*v2*dx(1)
            A -= self.S12M*phi1*v2*dx(1)
            F = (self.nusigf1M*phi1*v1 + self.nusigf2M*phi2*v1)*dx(1)
        if 2 in available_tags:
            A += self.D1F*ufl.inner(ufl.grad(phi1), ufl.grad(v1))*dx(2)
            A += self.D1F*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(2)
            A += (self.Sa1F + self.S12F)*phi1*v1*dx(2)
            A += self.D2F*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(2)
            A += self.Sa2F*phi2*v2*dx(2)
            A -= self.S12F*phi1*v2*dx(2)
            F += (self.nusigf1F*phi1*v1 + self.nusigf2F*phi2*v1)*dx(2)
        if 3 in available_tags:
            A += self.D13*ufl.inner(ufl.grad(phi1), ufl.grad(v1))*dx(3)
            A += self.D13*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(3)
            A += (self.Sa13 + self.S123)*phi1*v1*dx(3)
            A += self.D23*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(3)
            A += self.Sa23*phi2*v2*dx(3)
            A -= self.S123*phi1*v2*dx(3)
            F += (self.nusigf13*phi1*v1 + self.nusigf23*phi2*v1)*dx(3)

        for i in range(4,9):
            if i in available_tags:
                A += self.D14*ufl.inner(ufl.grad(phi1), ufl.grad(v1))*dx(i)
                A += self.D14*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(i)
                A += (self.Sa14 + self.S124)*phi1*v1*dx(i)
                A += self.D24*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(i)
                A += self.Sa24*phi2*v2*dx(i)
                A -= self.S124*phi1*v2*dx(i)
                F += (self.nusigf14*phi1*v1 + self.nusigf24*phi2*v1)*dx(i)
    
        if self.bord_cond == 'dir':
            if self.facet_tags is None:
                boundary_facets = mesh.locate_entities_boundary(
                    self.domain, self.domain.topology.dim-1,
                    lambda x: np.full(x.shape[1], True, dtype=bool))
            else:
                boundary_facets = self.facet_tags.find(self.G_HEX)

            dofs1 = fem.locate_dofs_topological(self.V.sub(
                0), self.domain.topology.dim-1, boundary_facets)
            dofs2 = fem.locate_dofs_topological(self.V.sub(
                1), self.domain.topology.dim-1, boundary_facets)

            zero = default_scalar_type(0)
            bc1 = fem.dirichletbc(zero, dofs1, self.V.sub(0))
            bc2 = fem.dirichletbc(zero, dofs2, self.V.sub(1))
            bcs = [bc1, bc2]

        elif self.bord_cond == 'neu':
            bcs = []
        elif self.bord_cond == 'rob':
            ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags) \
                if self.facet_tags is not None else ufl.ds
            cst_rob = 0.4692
            alfa1 = cst_rob/self.D1M
            alfa2 = cst_rob/self.D2M
            
            A += alfa1 * phi1 * v1 * ds(11) + alfa2 * phi2 * v2 * ds(11)
            bcs = []

        elif self.bord_cond == 'mixed':
            cst_rob = 0.4692
            alfa1 = -cst_rob/self.D1
            alfa2 = -cst_rob/self.D2
            ds = ufl.Measure("ds", domain=self.domain,
                             subdomain_data=self.facet_tags)
            for i in [1, 2, 7, 8, 9, 10, 11, 13]:
                A += alfa1 * phi1 * v1 * ds(i) + alfa2 * phi2 * v2 * ds(i)
            bcs = []
        else:
            raise ValueError(
                "Condición de borde inexistente: debe ser 'dir', 'neu' o 'rob' ")

        # ensamble del sistema
        a = fem_petsc.assemble_matrix(fem.form(A), bcs=bcs, diagonal=1e2)
        a.assemble()
        f = fem_petsc.assemble_matrix(fem.form(F), bcs=bcs, diagonal=1e-2)
        f.assemble()
        self.a = a

        v0 =a.createVecLeft()
        v0.set(1.0)
        if self.eigmethod == 'powerit':
            lam_k, x_k, k, res, lambs = gen_power_it_dynamic_momentum(f, a, v0, self.power_tol, self.ksp_tol, max_iter=1000)
            self.eigval = lam_k
            self.power_its = k
            self.vec = x_k
            self.power_res = res
            self.lambs = lambs

            phi = fem.Function(self.V)
            phi.x.array[:] = self.vec.array
            phi1, phi2 = phi.split()
            V0 = fem.functionspace(self.domain, ('CG', 1))

            self.phi1_proj = fem.Function(V0)
            self.phi1_proj.interpolate(fem.Expression(phi1, V0.element.interpolation_points()))

            self.phi2_proj = fem.Function(V0)
            self.phi2_proj.interpolate(fem.Expression(phi2, V0.element.interpolation_points()))
        if self.eigmethod == 'slepc':
            eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
            eigensolver.setDimensions(self.N_eig)
            eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

            st = SLEPc.ST().create(MPI.COMM_WORLD)
            st.setType(SLEPc.ST.Type.SINVERT)
            st.setShift(1.0)
            st.setFromOptions()
            eigensolver.setST(st)
            eigensolver.setOperators(a, f)
            eigensolver.setFromOptions()

            eigensolver.solve()

            self.vr, self.vi = a.getVecs()
            lam = eigensolver.getEigenpair(0, self.vr, self.vi)
            self.eigval = lam
            phi = fem.Function(self.V)
            phi.x.array[:] = self.vr.array

            phi1, phi2 = phi.split()
            V0 = fem.functionspace(self.domain, ("CG", 1))

            self.phi1_proj = fem.Function(V0)
            self.phi1_proj.interpolate(fem.Expression(
                phi1, V0.element.interpolation_points()))

            self.phi2_proj = fem.Function(V0)
            self.phi2_proj.interpolate(fem.Expression(
                phi2, V0.element.interpolation_points()))

    def phi_norms(self, num = 0):
        phi1_norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.phi1_proj,self.phi1_proj) * ufl.dx)))
        phi2_norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.phi2_proj,self.phi2_proj) * ufl.dx)))

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
