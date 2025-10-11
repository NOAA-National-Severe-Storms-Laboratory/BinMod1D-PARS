# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:17:50 2025

@author: edwin.dunnavan
"""
import numpy as np
import math
import scipy.special as scip

def init_rk(rk_order):
    
    # --- Butcher tableaux (a_ij, b_i, c_i) for RK1–RK4 ---
    tableaux = {
                1: dict(a=[[0]], # RK 1st order
                        b=[1], 
                        c=[0]),
                2: dict(a=[[0, 0], # RK 2nd order
                           [0.5, 0]], 
                        b=[0, 1], 
                        c=[0, 0.5]),
                3: dict(a=[[0, 0, 0], # RK 3rd order
                           [0.5, 0, 0],
                           [-1, 2, 0]], 
                        b=[1/6, 2/3, 1/6], 
                        c=[0, 0.5, 1]),
                4: dict(a=[[0, 0, 0, 0], # RK 4th order
                           [0.5, 0, 0, 0],
                           [0, 0.5, 0, 0],
                           [0, 0, 1, 0]], 
                        b=[1/6, 1/3, 1/3, 1/6], 
                        c=[0, 0.5, 0.5, 1])
                }

    if rk_order not in tableaux:
        raise ValueError("rk_order must be 1, 2, 3, or 4")

    RK = {'a':np.array(tableaux[rk_order]['a'], dtype=float),
          'b':np.array(tableaux[rk_order]['b'], dtype=float),
          'c':np.array(tableaux[rk_order]['c'], dtype=float)}
    
    return RK 

def kronecker_delta(i, j):
    return 1 if i == j else 0

def compute_coeffs(px, py, m, a, b, c, d):
    coeffs = {}
    for i in range(px, px + m + 2):       # possible x exponents
        for j in range(py, py + m + 2):   # possible y exponents
            Cij = 0

            # term with "a"
            k = i - px
            if 0 <= k <= m:
                Cij += a * math.comb(m, k) * kronecker_delta(j, py + m - k)

            # term with "b*x"
            k = i - px - 1
            if 0 <= k <= m:
                Cij += b * math.comb(m, k) * kronecker_delta(j, py + m - k)

            # term with "c*y"
            k = i - px
            if 0 <= k <= m:
                Cij += c * math.comb(m, k) * kronecker_delta(j, py + m - k + 1)

            # term with "d*x*y"
            k = i - px - 1
            if 0 <= k <= m:
                Cij += d * math.comb(m, k) * kronecker_delta(j, py + m - k + 1)

            if Cij != 0:
                coeffs[(i, j)] = Cij

    return coeffs

def compute_combined_coeffs_NxNy(px, py, m, a, b, c, d, ax, cx, ay, cy):
    """Return coefficients of F(x,y)*Nx(x)*Ny(y)."""
    coeffs_F = compute_coeffs(px, py, m, a, b, c, d)
    coeffs_F_N = {}
    for (i, j), coeff in coeffs_F.items():
        # a_x*a_y * Cij * x^(i+1) y^(j+1)
        coeffs_F_N[(i+1, j+1)] = coeffs_F_N.get((i+1, j+1), 0) + ax * ay * coeff
        # a_x*c_y * Cij * x^(i+1) y^j
        coeffs_F_N[(i+1, j)] = coeffs_F_N.get((i+1, j), 0) + ax * cy * coeff
        # c_x*a_y * Cij * x^i y^(j+1)
        coeffs_F_N[(i, j+1)] = coeffs_F_N.get((i, j+1), 0) + cx * ay * coeff
        # c_x*c_y * Cij * x^i y^j
        coeffs_F_N[(i, j)] = coeffs_F_N.get((i, j), 0) + cx * cy * coeff
    return coeffs_F_N

def compute_coeffs_vectorized(px, py, m, a, b, c, d):
    """
    Vectorized computation of coefficients C[i,j] for all a,b,c,d arrays of shape (M,N).
    Returns a dict of {(i,j): array of shape (M,N)}.
    """
    #M, N = a.shape
    coeffs = {}

    for i in range(px, px + m + 2):
        for j in range(py, py + m + 2):
            Cij = np.zeros_like(a, dtype=float)

            # term with "a"
            k = i - px
            if 0 <= k <= m and j == py + m - k:
                Cij += a * math.comb(m, k)

            # term with "b*x"
            k = i - px - 1
            if 0 <= k <= m and j == py + m - k:
                Cij += b * math.comb(m, k)

            # term with "c*y"
            k = i - px
            if 0 <= k <= m and j == py + m - k + 1:
                Cij += c * math.comb(m, k)

            # term with "d*x*y"
            k = i - px - 1
            if 0 <= k <= m and j == py + m - k + 1:
                Cij += d * math.comb(m, k)

            if not np.all(Cij == 0):
                coeffs[(i, j)] = Cij

    return coeffs

def compute_combined_coeffs_NxNy_vectorized(px, py, m, a, b, c, d, ax, cx, ay, cy):
    """
    Compute coefficients of F(x,y)*Nx(x)*Ny(y) with Nx(x)=ax*x+cx, Ny(y)=ay*y+cy.
    Returns dict {(i,j): array of shape (M,N)}.
    """
    coeffs_F = compute_coeffs_vectorized(px, py, m, a, b, c, d)
    coeffs_F_N = {}

    for (i, j), Cij in coeffs_F.items():
        # ax*ay * Cij * x^(i+1) y^(j+1)
        coeffs_F_N[(i+1, j+1)] = coeffs_F_N.get((i+1, j+1), 0) + ax * ay * Cij
        # ax*cy * Cij * x^(i+1) y^j
        coeffs_F_N[(i+1, j)] = coeffs_F_N.get((i+1, j), 0) + ax * cy * Cij
        # cx*ay * Cij * x^i y^(j+1)
        coeffs_F_N[(i, j+1)] = coeffs_F_N.get((i, j+1), 0) + cx * ay * Cij
        # cx*cy * Cij * x^i y^j
        coeffs_F_N[(i, j)] = coeffs_F_N.get((i, j), 0) + cx * cy * Cij

    return coeffs_F_N

def triangle_monomial_integral(i, j, xt1, yt1, xt2, yt2, xt3, yt3):
    """
    Vectorized integral of x^i y^j over many triangles defined by vertices arrays.
    All xt1, yt1, ... xt3, yt3 have same shape.
    Returns array of same shape.
    """
    dx2 = xt2 - xt1
    dy2 = yt2 - yt1
    dx3 = xt3 - xt1
    dy3 = yt3 - yt1
    J = np.abs(dx2 * dy3 - dx3 * dy2)  # Jacobian determinant

    # Expand (x1 + u*dx2 + v*dx3)^i (y1 + u*dy2 + v*dy3)^j
    integral = np.zeros_like(xt1, dtype=float)
    for p in range(i+1):
        for q in range(j+1):
            coeff = math.comb(i, p) * math.comb(j, q) * (xt1**(i-p)) * (yt1**(j-q))
            for r in range(p+1):
                for s in range(q+1):
                    upow = r + s
                    vpow = (p-r) + (q-s)
                    coeff_uv = coeff * math.comb(p, r) * math.comb(q, s) * \
                               (dx2**r) * (dy2**s) * (dx3**(p-r)) * (dy3**(q-s))
                    Iuv = math.factorial(upow) * math.factorial(vpow) / math.factorial(upow + vpow + 2)
                    integral += coeff_uv * Iuv
    return J * integral


def integrate_rect_kernel(x1, x2, y1, y2, px, py, m, f, ax, cx, ay, cy):
    '''
    Integral of: x^px * y^py *(x+y)^m * [K(x,y)=a+b*x+c*y+d*x*y]*[n1(x)=ax*x+cx]*[n2(y)=ay*y+cy] 
    '''
    a = f[0,:] 
    b = f[1,:] 
    c = f[2,:] 
    d = f[3,:]
    
    total = np.zeros_like(a)
    
    coeffs_F_N = compute_combined_coeffs_NxNy_vectorized(px, py, m, a, b, c, d, ax, cx, ay, cy)
 
    for (i, j), coeff in coeffs_F_N.items():
        total += coeff * Pn(i,x1,x2) * Pn(j,y1,y2)  
              
    return total

def integrate_tri_kernel(px, py, m, f,
                               ax, cx, ay, cy,
                               xt1, yt1, xt2, yt2, xt3, yt3):
    
    a = f[0,:] 
    b = f[1,:] 
    c = f[2,:] 
    d = f[3,:]
    
    coeffs_F_N = compute_combined_coeffs_NxNy_vectorized(px, py, m, a, b, c, d,
                                                         ax, cx, ay, cy)
    total = np.zeros_like(a, dtype=float)
    
    for (i, j), Cij in coeffs_F_N.items():
        Iij = triangle_monomial_integral(i, j, xt1, yt1, xt2, yt2, xt3, yt3)
        total += Cij * Iij
        
    return total

def LGN_int(n,muf,sig2f,x1,x2):
    # 

    #I = 0.5*np.exp(0.5*n*(n-1)*sig2f)*\
   #     (scip.erf((np.log(x2-muf)-sig2f*(n-0.5))/(np.sqrt(2*sig2f)))-\
   #      scip.erf((np.log(x1-muf)-sig2f*(n-0.5))/(np.sqrt(2*sig2f))))

    I = 0.5*np.exp(n*muf+0.5*n**2*sig2f)*\
        (scip.erf((np.log(x2)-muf-n*sig2f)/(np.sqrt(2*sig2f)))-\
         scip.erf((np.log(x1)-muf-n*sig2f)/(np.sqrt(2*sig2f))))


    return I 

def LGN_int_PB07(n,muf,sig2f,x1,x2):
    # moments of lognormal fragment mass distribution from x1 to x2
    # See Prat and Barros (2007)
    
    n += 1

    t1 = np.log(x1-muf)/np.sqrt(sig2f)
    t2 = np.log(x2-muf)/np.sqrt(sig2f)

    I = (np.sqrt(np.pi*sig2f)/2.)*np.exp(n*(muf+0.25*n*sig2f))*\
        (scip.erf(t2-0.5*n*np.sqrt(sig2f))-scip.erf(t1-0.5*n*np.sqrt(sig2f)))
         
    # NORMALIZE I so that integral of mass from x1[0] to x2[-1] evaluates to unity.
    # This is needed to preserve the total mass from each interaction.

    return I 


def gam_int(n,mu,Dm,x1,x2):
    # integral of x^n *x^mu * exp(-c*x) from x1 to x2
    
    #I = c**(-n)*(scip.gammainc(n+mu+1.,c*x2)-scip.gammainc(n+mu+1.,c*x1))
    
    I = (mu+4)**(-n)*Dm**n*scip.gamma(n+mu+1)*(scip.gammainc(n+mu+1.,(mu+4)*x2/Dm)-scip.gammainc(n+mu+1.,(mu+4)*x1/Dm))
    
    return I    

def In_int(n,c,x1,x2):
    # integral of x^n * exp(-c*x) from x1 to x2
    
    if n==0:
        
        I = (1./c)*(np.exp(-c*x1)-np.exp(-c*x2))
        
    elif n==1:
        
        I = (1./c**2)*(np.exp(-c*x1)*(c*x1+1.)-np.exp(-c*x2)*(c*x2+1.))
        
    else:
        
        print('n needs to be either 0 or 1.')
        raise Exception()
    
    return I    


def In_int_old(n,c,x1,x2):
    # integral of x^n * exp(-c*x) from x1 to x2
    if c != 0:
        m = np.arange(0,2)
        i = np.arange(0,n+1,dtype='int')
        xm = np.array((x1,x2))
    
        I = 0.
    
        for mm in m:
            Itemp = 0.
            
            for ii in i:
                Itemp +=(xm[mm]**ii)/(scip.factorial(ii)*c**(n-ii+1))
        
            I += (-1)**(mm)*np.exp(-c*xm[mm])*Itemp
    
        I = scip.factorial(n)*I
    elif c ==0:
        I = Pn(n,x1,x2)

    return I    

def Pn(n,x1,x2):
    # integral of x^n from x1 to x2
    
    n = n+1
    
    if n <= 1:
        P = (x2 - x1)
    else:
        P = (x2**n - x1**n)/n
    
    return P