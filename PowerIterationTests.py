from petsc4py import PETSc
import numpy as np
from petsc4py import PETSc
import numpy as np

def gen_power_it_dynamic_momentum(A, B, v0, tol=1e-8, max_iter=1000):
    '''
    Dynamic power itetration with momentum to solver the generalized eigenvalue problem Ax = lambda Bx

    Input:
    - A  petc.mat
    - B, spd  petc.mat
    - v0 petc.vec

    Output:
    - lambda: largest eigenvalue,  multiplication factor
    - x: eigenvector corresponding to the largest eigenvalue
    - k: number of iterations
    - res: residuals list
    '''
    res = []
    
    v   = B.createVecRight()
    rhs = A.createVecLeft()
    Ax = A.createVecLeft()
    Bx = B.createVecLeft()
    #r    = A.createVecLeft()

    def Bnorm(x):
        y = B.createVecLeft()
        B.mult(x, y)
        q = x.dot(y)                       
        return float(np.sqrt(q))

    # comm = A.getComm()
    # N = B.getSize()[0]
    # n0 = N // 2
    # n1 = N - n0

    # # IS contiguos
    # is0 = PETSc.IS().createStride(n0, first=0,  step=1, comm=comm)
    # print(is0)
    # is1 = PETSc.IS().createStride(n1, first=n0, step=1, comm=comm)

    # # Submatrices
    # A00 = A.createSubMatrix(is0, is0)
    # A10 = A.createSubMatrix(is1, is0)
    # A11 = A.createSubMatrix(is1, is1)


    # # KSPs
    # ksp0 = PETSc.KSP().create(A00.getComm())
    # ksp0.setOperators(A00)
    # ksp0.setType('gmres'); ksp0.getPC().setType('gamg')
    # ksp0.setTolerances(rtol=1e-8, max_it=500)
    # ksp0.setFromOptions(); ksp0.setUp()

    # ksp1 = PETSc.KSP().create(A11.getComm())
    # ksp1.setOperators(A11)
    # ksp1.setType('gmres'); ksp1.getPC().setType('gamg')
    # ksp1.setTolerances(rtol=1e-8, max_it=500)
    # ksp1.setFromOptions(); ksp1.setUp()

    def solve_block_lower_half(A: PETSc.Mat, b: PETSc.Vec, n0=None):
        """
        Resuelve L x = b con L = [[A00, 0], [A10, A11]]
        suponiendo que los bloques son mitades contiguas.
        Devuelve x (Vec global).
        """
        comm = A.getComm()
        N = B.getSize()[0]
        n0 = N // 2
        n1 = N - n0

        # IS contiguos
        is0 = PETSc.IS().createStride(n0, first=0,  step=1, comm=comm)
        is1 = PETSc.IS().createStride(n1, first=n0, step=1, comm=comm)

        # Submatrices
        A00 = A.createSubMatrix(is0, is0)
        A10 = A.createSubMatrix(is1, is0)
        A11 = A.createSubMatrix(is1, is1)


        # KSPs
        ksp0 = PETSc.KSP().create(A00.getComm())
        ksp0.setOperators(A00)
        ksp0.setType('gmres'); ksp0.getPC().setType('gamg')
        ksp0.setTolerances(rtol=1e-8, max_it=500)
        ksp0.setFromOptions(); ksp0.setUp()

        ksp1 = PETSc.KSP().create(A11.getComm())
        ksp1.setOperators(A11)
        ksp1.setType('gmres'); ksp1.getPC().setType('gamg')
        ksp1.setTolerances(rtol=1e-8, max_it=500)
        ksp1.setFromOptions(); ksp1.setUp()
        # b0, b1
        b0 = b.getSubVector(is0)
        b1 = b.getSubVector(is1)

        # A00 x0 = b0
        x0 = b0.duplicate()
        ksp0.solve(b0, x0)

        # A11x1 = b1 - A10x0
        r1 = b1.duplicate()
        A10.mult(x0, r1)
        r1.axpy(-1.0, b1)
        x1 = b1.duplicate()
        ksp1.solve(r1, x1)        
    
        #rearmar
        #---------------------------------------------------------------------------------------------------------
        #Aca se rearma el vector pero no se qué está pasando
        x = b.duplicate(); x.set(0.0)

        # from x0 (todos sus entries) -> to x[is0]
        scat0 = PETSc.Scatter().create(x0, None, x, is0)
        scat0.scatter(x0, x, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # from x1 -> to x[is1]
        scat1 = PETSc.Scatter().create(x1, None, x, is1)
        scat1.scatter(x1, x, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Restaurar vistas
        b.restoreSubVector(is0, b0)
        b.restoreSubVector(is1, b1)
        #---------------------------------------------------------------------------------------------------------------
        #x = b.duplicate(); x.set(0.0)
        #x, [is0, is1] = PETSc.Vec.concatenate([x0, x1])
        return x


    # Inicialización
    h0 = Bnorm(v0)
    x0 = v0.copy()
    x0.scale(1.0/h0)
    #rhs = A.createVecLeft()
    A.mult(x0, rhs)    
    #v   = B.createVecRight()     
    v1_vec = solve_block_lower_half(B,rhs)

    # k = 0 ----------------------------------

    # h1
    h1 = Bnorm(v1_vec)

    # x1
    x_km1 = v1_vec.copy()
    x_km1.scale(1.0/h1)

    # lambda1
    #Ax = A.createVecLeft()
    A.mult(x_km1, Ax)
    Bx = B.createVecLeft()
    B.mult(x_km1, Bx)
    lam1 = float((x_km1.dot(Ax)))

    #d1
    r = Ax.copy()
    #r.copy(Ax)
    r.axpy(-lam1, Bx)
    d_prev = r.norm()
    res.append(d_prev)

    if d_prev < tol: 
        return lam1, x_km1, 1, res

    # v1
    #rhs = A.createVecLeft()
    A.mult(x_km1, rhs)
    v2_vec = solve_block_lower_half(B,rhs)

    # k = 1 --------------------------------

    #h2
    h_k = Bnorm(v2_vec)
    x_k = v2_vec.copy()
    x_k.scale(1.0/h_k)

    # lambda 2
    #Ax = A.createVecLeft()
    A.mult(x_k, Ax)
    #Bx = B.createVecLeft()
    B.mult(x_k, Bx)
    lam_k = float((x_k.dot(Ax)))

    #d2 residual
    r = Ax.copy()
    #r.copy(Ax)
    r.axpy(-lam_k, Bx)
    d_k = r.norm()
    res.append(d_k)
    if d_k < tol: 
        return lam_k, x_k, 2, res

    # redefine parameters
    r_k = min(d_k/d_prev, 1.0)
    k = 2

    # k >= 2 --------------------------------------------------

    while k < max_iter:

        # beta_k
        beta_k = (lam_k**2) * (r_k**2) / 4.0

        #v_{k+1}
        #rhs = A.createVecLeft()
        A.mult(x_k, rhs)
        v_kp1 = solve_block_lower_half(B,rhs) 

        # u_{k+1}
        u_kp1 = v_kp1.copy()
        u_kp1.axpy(-(beta_k / h_k), x_km1)

        # h_{k+1}
        h_kp1 = Bnorm(u_kp1)

        # x_{k+1}
        x_kp1 = u_kp1.copy()
        x_kp1.scale(1.0/h_kp1)

        # lambda_{k+1}
        #Ax = A.createVecLeft()
        A.mult(x_kp1, Ax)
        #Bx = B.createVecLeft()
        B.mult(x_kp1, Bx)
        lam_kp1 = float((x_kp1.dot(Ax)))

        # d_{k+1}
        r = Ax.copy()
        #r.copy(Ax)
        r.axpy(-lam_kp1, Bx)
        d_kp1 = r.norm()
        res.append(d_kp1)
        if d_kp1 < tol: 
            return lam_kp1, x_kp1, k+1, res

        # rho_k
        rho_k = min(d_k/d_prev, 1.0)

        #r_{k+1}
        r_k = 2.0*rho_k/(1.0 + rho_k**2)

        #Update parameters
        x_km1 = x_k
        x_k = x_kp1
        h_k = h_kp1
        d_prev = d_k
        d_k =  d_kp1
        lam_k = lam_kp1
        k += 1

    return lam_k, x_k, k, res