from petsc4py import PETSc
import numpy as np
from petsc4py import PETSc
import numpy as np

# Aux functions
def Bnorm(x, _work):
    B.mult(x, _work)
    q = x.dot(_work)
    return float(np.sqrt(q))

def solve_B(rhs, sol):
    '''
    Preconditioned linear solver
    '''
    ksp.solve(rhs, sol)

def update_residual(A, Ax, B, Bx, x, r):
    A.mult(x, Ax)
    B.mult(x, Bx)
    lam = float((x.dot(Ax)))
    Ax.copy(r)
    r.axpy(-lam, Bx)
    return lam


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
    h = Bnorm(B, x, work)
    vec.copy(out)
    out.scale(1.0/h)
    return h



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

    # Initialize ksp
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(B)
    ksp.setType('gmres')
    ksp.getPC().setType('gamg')
    ksp.setTolerances(rtol=1e-8, max_it =1000)
    ksp.setFromOptions()
    ksp.getIterationNumber()
    ksp.setUp()

    # Init vectors
    v   = B.createVecRight()
    rhs = A.createVecLeft()
    Ax = A.createVecLeft()
    Bx = B.createVecLeft()
    v_kp1 = A.createVecLeft()
    u_kp1 = v_kp1.copy()
    x_kp1 = u_kp1.copy()
    v1_vec = A.createVecLeft()
    v2_vec = A.createVecLeft()
    r    = A.createVecLeft()
    work = B.createVecLeft()
    

    # First power iteration
    # Inicialización
    x0 = v0.copy()
    h0 = Bnormalize(B, v0, x0, work)
    apply_operator(ksp, A, x0, rhs, v1_vec)

    # NB: Acá no está el control de error. Me parece bien, pero mejor dejarlo comentado:
    # We skip the first error control to avoid unnecessary computations.

    # ------ k = 0 -------
    # h1
    x1 = v1_vec.copy()
    h1 = Bnorm(v1_vec, work)

    # NB: Acá hay algo raro. No usas x1 después? 
    # Dejo comentado lo que creo que debería pasar:
    # h1 = Bnormalize(B, v1_vec, x1, work)
    # apply_operator(ksp, A, x1, rhs, v2_vec)


    # x1
    x_km1 = v1_vec.copy()
    x_km1.scale(1.0/h1)

    # lambda1
    lam1 = update_residual(A, Ax, B, Bx, x_km1, r)
    d_prev = r.norm()
    res.append(d_prev)

    if d_prev < tol: 
        return lam1, x_km1, 1, res

    # v1
    apply_operator(ksp, A, x_km1, rhs, v2_vec)
    # NB: Mi código reemplaza hasta acá. Imagino que el control de 
    # error viene después de aplicar A. 

    # k = 1 --------------------------------

    # NB: Acá se puede hacer lo mismo en términos de Bnormalize y 
    # apply_operator. Creo que es más limpio.
    #h2
    h_k = Bnorm(v2_vec, work)
    x_k = v2_vec.copy()
    x_k.scale(1.0/h_k)

    # lambda 2
    lam_k = update_residual(A, Ax, B, Bx, x_k, r)
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
        apply_operator(ksp, A, x_k, rhs, v_kp1)

        # u_{k+1}
        v_kp1.copy(u_kp1)
        u_kp1.axpy(-(beta_k / h_k), x_km1) # NB: x_km1, no x_k?

        # h_{k+1}
        B.mult(u_kp1, y)
        q = u_kp1.dot(y)
        h_kp1 = float(np.sqrt(q))

        # x_{k+1}
        u_kp1.copy(x_kp1)
        x_kp1.scale(1.0/h_kp1)

        # lambda_{k+1}
        lam_kp1 = update_residual(A, Ax, B, Bx, x_kp1, r)

        # d_{k+1}
        d_kp1 = r.norm()
        res.append(d_kp1)
        if d_kp1 < tol: 
            return lam_kp1, x_kp1, k+1, res

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

    return lam_k, x_k, k, res

# def gen_power_it_dynamic_momentum(A, B, v0, tol=1e-8, max_iter=1000):
#     '''
#     Dynamic power itetration with momentum to solver the generalized eigenvalue problem Ax = lambda Bx

#     Input:
#     - A  petc.mat
#     - B, spd  petc.mat
#     - v0 petc.vec

#     Output:
#     - lambda: largest eigenvalue,  multiplication factor
#     - x: eigenvector corresponding to the largest eigenvalue
#     - k: number of iterations
#     - res: residuals list
#     '''
#     res = []

#     ksp = PETSc.KSP().create(A.getComm())
#     ksp.setOperators(B)
#     ksp.setType('cg')
#     ksp.getPC().setType('gamg')
#     ksp.setTolerances(rtol=1e-8, max_it =1000)
#     ksp.setFromOptions()
#     ksp.setUp()

#     v   = B.createVecRight()
#     rhs = A.createVecLeft()
#     Ax = A.createVecLeft()
#     Bx = B.createVecLeft()
#     #r    = A.createVecLeft()

#     def Bnorm(x):
#         y = B.createVecLeft()
#         B.mult(x, y)
#         q = x.dot(y)                       
#         return float(np.sqrt(q))

    
#     def solve_B(rhs, y):
#         '''
#         Preconditioned lineal solver
#         '''
#         y.set(0.0)
#         ksp.solve(rhs, y)
#         return y

#     # Inicialización
#     h0 = Bnorm(v0)
#     x0 = v0.copy()
#     x0.scale(1.0/h0)
#     #rhs = A.createVecLeft()
#     A.mult(x0, rhs)    
#     #v   = B.createVecRight()     
#     v1_vec = solve_B(rhs,v)

#     # k = 0 ----------------------------------

#     # h1
#     h1 = Bnorm(v1_vec)

#     # x1
#     x_km1 = v1_vec.copy()
#     x_km1.scale(1.0/h1)

#     # lambda1
#     #Ax = A.createVecLeft()
#     A.mult(x_km1, Ax)
#     Bx = B.createVecLeft()
#     B.mult(x_km1, Bx)
#     lam1 = float((x_km1.dot(Ax)/x_km1.dot(Bx)))

#     #d1
#     r = Ax.copy()
#     #r.copy(Ax)
#     r.axpy(-lam1, Bx)
#     d_prev = r.norm()
#     res.append(d_prev)

#     if d_prev < tol: 
#         return lam1, x_km1, 1, res

#     # v1
#     #rhs = A.createVecLeft()
#     A.mult(x_km1, rhs)
#     v2_vec = solve_B(rhs,v)

#     # k = 1 --------------------------------

#     #h2
#     h_k = Bnorm(v2_vec)
#     x_k = v2_vec.copy()
#     x_k.scale(1.0/h_k)

#     # lambda 2
#     #Ax = A.createVecLeft()
#     A.mult(x_k, Ax)
#     #Bx = B.createVecLeft()
#     B.mult(x_k, Bx)
#     lam_k = float((x_k.dot(Ax)/x_k.dot(Bx)))

#     #d2 residual
#     r = Ax.copy()
#     #r.copy(Ax)
#     r.axpy(-lam_k, Bx)
#     d_k = r.norm()
#     res.append(d_k)
#     if d_k < tol: 
#         return lam_k, x_k, 2, res

#     # redefine parameters
#     r_k = min(d_k/d_prev, 1.0)
#     k = 2

#     # k >= 2 --------------------------------------------------

#     while k < max_iter:

#         # beta_k
#         beta_k = (lam_k**2) * (r_k**2) / 4.0

#         #v_{k+1}
#         #rhs = A.createVecLeft()
#         A.mult(x_k, rhs)
#         v_kp1 = solve_B(rhs,v) 

#         # u_{k+1}
#         u_kp1 = v_kp1.copy()
#         u_kp1.axpy(-(beta_k / h_k), x_km1)

#         # h_{k+1}
#         h_kp1 = Bnorm(u_kp1)

#         # x_{k+1}
#         x_kp1 = u_kp1.copy()
#         x_kp1.scale(1.0/h_kp1)

#         # lambda_{k+1}
#         #Ax = A.createVecLeft()
#         A.mult(x_kp1, Ax)
#         #Bx = B.createVecLeft()
#         B.mult(x_kp1, Bx)
#         lam_kp1 = float((x_kp1.dot(Ax) / x_kp1.dot(Bx)))

#         # d_{k+1}
#         r = Ax.copy()
#         #r.copy(Ax)
#         r.axpy(-lam_kp1, Bx)
#         d_kp1 = r.norm()
#         res.append(d_kp1)
#         if d_kp1 < tol: 
#             return lam_kp1, x_kp1, k+1, res

#         # rho_k
#         rho_k = min(d_k/d_prev, 1.0)

#         #r_{k+1}
#         r_k = 2.0*rho_k/(1.0 + rho_k**2)

#         #Update parameters
#         x_km1 = x_k
#         x_k = x_kp1
#         h_k = h_kp1
#         d_prev = d_k
#         d_k =  d_kp1
#         lam_k = lam_kp1
#         k += 1

#     return lam_k, x_k, k, res

# def gen_power_it_dynamic_momentum(A, B, v0, tol=1e-8, max_iter=1000):
#     '''
#     Dynamic power itetration with momentum to solver the generalized eigenvalue problem Ax = lambda Bx

#     Input:
#     - A  petc.mat
#     - B, spd  petc.mat
#     - v0 petc.vec

#     Output:
#     - lambda: largest eigenvalue,  multiplication factor
#     - x: eigenvector corresponding to the largest eigenvalue
#     - k: number of iterations
#     - res: residuals list
#     '''
#     res = []

#     def Bnorm(x):
#         y = B.createVecLeft()
#         B.mult(x, y)
#         q = x.dot(y)                       
#         return float(np.sqrt(q))

    
#     def solve_B(rhs):
#         '''
#         Preconditioned lineal solver
#         '''
#         y = B.createVecRight()
#         y.set(0.0)
#         ksp = PETSc.KSP().create()
#         ksp.setOperators(B)
#         ksp.setType('cg')
#         ksp.getPC().setType('gamg')
#         ksp.setTolerances(rtol=1e-8, max_it =1000)
#         ksp.setFromOptions()
#         ksp.solve(rhs, y)
#         return y

#     # Inicialización
#     h0 = Bnorm(v0)
#     x0 = v0.copy()
#     x0.scale(1.0/h0)

#     rhs = A.createVecLeft()
#     A.mult(x0, rhs)         
#     v1_vec = solve_B(rhs)

#     # k = 0 ----------------------------------

#     # h1
#     h1 = Bnorm(v1_vec)

#     # x1
#     x1 = v1_vec.copy()
#     x1.scale(1.0/h1)

#     # lambda1
#     Ax = A.createVecLeft()
#     A.mult(x1, Ax)
#     Bx = B.createVecLeft()
#     B.mult(x1, Bx)
#     lam1 = float((x1.dot(Ax)/x1.dot(Bx)))

#     #d1
#     r = Ax.copy()
#     r.axpy(-lam1, Bx)
#     d1 = r.norm()
#     res.append(d1)

#     if d1 < tol: 
#         return lam1, x1, 1, res

#     # v1
#     rhs = A.createVecLeft()
#     A.mult(x1, rhs)
#     v2_vec = solve_B(rhs)

#     # k = 1 --------------------------------

#     #h2
#     h2 = Bnorm(v2_vec)
#     x2 = v2_vec.copy()
#     x2.scale(1.0/h2)

#     # lambda 2
#     Ax = A.createVecLeft()
#     A.mult(x2, Ax)
#     Bx = B.createVecLeft()
#     B.mult(x2, Bx)
#     lam2 = float((x2.dot(Ax)/x2.dot(Bx)))

#     #d2 residual
#     r = Ax.copy()
#     r.axpy(-lam2, Bx)
#     d2 = r.norm()
#     res.append(d2)
#     if d2 < tol: 
#         return lam2, x2, 2, res

#     # redefine parameters
#     r_k = min(d2/d1, 1.0)
#     k = 2
#     x_km1, x_k = x1.copy(), x2.copy()
#     h_k, lam_k = h2, lam2
#     d_prev, d_k = d1, d2

#     # k >= 2 --------------------------------------------------

#     while k < max_iter:

#         # beta_k
#         beta_k = (lam_k**2) * (r_k**2) / 4.0

#         #v_{k+1}
#         rhs = A.createVecLeft()
#         A.mult(x_k, rhs)
#         v_kp1 = solve_B(rhs) 

#         # u_{k+1}
#         u_kp1 = v_kp1.copy()
#         u_kp1.axpy(-(beta_k / h_k), x_km1)

#         # h_{k+1}
#         h_kp1 = Bnorm(u_kp1)

#         # x_{k+1}
#         x_kp1 = u_kp1.copy()
#         x_kp1.scale(1.0/h_kp1)

#         # lambda_{k+1}
#         Ax = A.createVecLeft()
#         A.mult(x_kp1, Ax)
#         Bx = B.createVecLeft()
#         B.mult(x_kp1, Bx)
#         lam_kp1 = float((x_kp1.dot(Ax) / x_kp1.dot(Bx)))

#         # d_{k+1}
#         r = Ax.copy()
#         r.axpy(-lam_kp1, Bx)
#         d_kp1 = r.norm()
#         res.append(d_kp1)
#         if d_kp1 < tol: 
#             return lam_kp1, x_kp1, k+1, res

#         # rho_k
#         rho_k = min(d_k/d_prev, 1.0)

#         #r_{k+1}
#         r_k = 2.0*rho_k/(1.0 + rho_k**2)

#         #Update parameters
#         x_km1 = x_k
#         x_k = x_kp1
#         h_k = h_kp1
#         d_prev = d_k
#         d_k =  d_kp1
#         lam_k = lam_kp1
#         k += 1

#     return lam_k, x_k, k, res
