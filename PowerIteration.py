from petsc4py import PETSc
import numpy as np
from petsc4py import PETSc
import numpy as np

# Aux functions
def Bnorm(B, x, _work):
    B.mult(x, _work)
    q = x.dot(_work)
    return float(np.sqrt(q))

def solve_B(rhs, sol):
    '''
    Preconditioned linear solver
    '''
    ksp.solve(rhs, sol)

def update_residual(A, B, x, r, Ax_work):
    A.mult(x, Ax_work)
    lam = float((x.dot(Ax_work)))
    Ax_work.copy(r) # r = Ax
    B.mult(x, Ax_work)
    r.axpy(-lam, Ax_work) # r = Ax - lam Bx
    err_norm = r.norm()
    return lam, err_norm


def apply_operator(ksp, A, x, rhs, sol):
    """
    Compute sol = B^{-1} A x
    """
    A.mult(x, rhs)
    ksp.solve(rhs, sol)

def Bnormalize(B, vec, out, work):
    """
    compute h = |vec|_B and out = 1/h vec
    work vector is used internally for computations.
    """
    h = Bnorm(B, vec, work)
    vec.copy(out)
    out.scale(1.0/h)
    return h



def gen_power_it_dynamic_momentum(A, B, v0, tol, ksp_tol, max_iter=1000):
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
    lambs =[]

    # Initialize ksp
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(B)
    ksp.setType('gmres')
    ksp.getPC().setType('hypre')
    ksp.setTolerances(rtol=ksp_tol, max_it =1000)
    ksp.setFromOptions()
    ksp.getIterationNumber()
    ksp.setUp()

    # Init vectors
    x_kp1 = v0.copy()
    x_k   = v0.copy()
    x_km1 = v0.copy()
    v_kp1 = A.createVecLeft()
    r    = A.createVecLeft()
    work = B.createVecLeft()
    Ax_work = A.createVecLeft()
    rhs_work = work

    # First power iteration
    # Inicialización
    x0 = x_k # x_k is then replaced
    v1_vec = v_kp1
    h0 = Bnormalize(B, v0, x0, work)
    apply_operator(ksp, A, x0, rhs_work, v1_vec)

    # ------ k = 0 -------

    # h1, x1
    v2_vec = v_kp1
    h1 = Bnormalize(B, v1_vec, x_km1, work)
    apply_operator(ksp, A, x_km1, rhs_work, v2_vec)

    # lambda1
    lam1, d_prev = update_residual(A, B, x_km1, r, Ax_work)
    res.append(d_prev)
    lambs.append(lam1)
    if d_prev < tol: 
        return lam1, x_km1, 1, res, lambs


    # k = 1 --------------------------------

    # NB: Acá se puede hacer lo mismo en términos de Bnormalize y 
    # apply_operator. Creo que es más limpio.
    #h2
    h_k = Bnormalize(B, v2_vec, x_k, work)

    # lambda 2
    lam_k, d_k = update_residual(A, B, x_k, r, Ax_work)
    lambs.append(lam_k)
    res.append(d_k)
    if d_k < tol: 
        return lam_k, x_k, 2, res, lambs

    # redefine parameters
    r_k = min(d_k/d_prev, 1.0)
    k = 2

    # k >= 2 --------------------------------------------------
    itersnum = []
    while k < max_iter:

        # beta_k
        beta_k = (lam_k**2) * (r_k**2) / 4.0

        #v_{k+1}
        apply_operator(ksp, A, x_k, rhs_work, v_kp1)
        itersnum.append(ksp.getIterationNumber())

        # u_{k+1}
        u_kp1 = Ax_work # Some generic, unused work vector
        v_kp1.copy(u_kp1)
        u_kp1.axpy(-(beta_k / h_k), x_km1)

        # h_{k+1}
        h_kp1 = Bnormalize(B, u_kp1, x_kp1, work)

        # lambda_{k+1}
        lam_kp1, d_kp1 = update_residual(A, B, x_kp1, r, Ax_work)
        lambs.append(lam_kp1)
        # d_{k+1}
        res.append(d_kp1)
        if d_kp1 < tol: 
            print('prom iteraciones gmres: ', sum(itersnum)/len(itersnum))
            return lam_kp1, x_kp1, k+1, res, lambs

        # rho_k
        rho_k = min(d_k/d_prev, 1.0)

        #r_{k+1}
        r_k = 2.0*rho_k/(1.0 + rho_k**2)

        #Update parameters
        x_k.copy(x_km1)
        x_kp1.copy(x_k)
        h_k = h_kp1
        d_prev = d_k
        d_k =  d_kp1
        lam_k = lam_kp1
        k += 1
    print('prom iteraciones gmres: ', sum(itersnum)/len(itersnum))
    return lam_k, x_k, k, res, lambs

