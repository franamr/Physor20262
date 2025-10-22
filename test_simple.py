from NeutronTransportSolver import NeutronTransportSolver
from dolfinx import mesh, fem, default_scalar_type
from mpi4py import MPI
import numpy as np
import time
print('Dynamic momentum power method')
print('------------------------------------------------------------------------------')

def solveProblem(N, dim, method):
    print(f'{N} x {N} Unit square')
    # Crear el dominio
    if dim == 2:
        domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    else:
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)

    # Crear el solver y resolver
    solver = NeutronTransportSolver(domain, bord_cond = 'rob', power_tol=1e-6, ksp_tol=1e-7, eigmethod=method)
    time0 = time.time()
    solver.solve()
    time1 = time.time()

    print('k_eff                                                   :', solver.eigval)
    if method == "powerit": print('Number of iterations Dynamic Power Method with momentum :', solver.power_its)
    print('CPU time                                                :', time1 - time0)

#Ns = [16, 32, 64, 128, 256, 512]
dim = 3
Ns = [16, 32, 64]
method = "slepc"
for N in Ns: 
    solveProblem(N, dim, method)


