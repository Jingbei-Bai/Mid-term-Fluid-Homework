
import sympy as sp

# Symbolic verification code for Question 1
# Scheme: u_j^{n+1} = sum_{k=l}^r b_k u_{j+k}^n
# Moment conditions for p-th order accuracy:
#   sum b_k k^m = (-lambda)^m,  m=0,1,...,p

lam = sp.symbols('lambda', real=True)


def verify_upwind():
    # First-order upwind scheme for a>0:
    # u_j^{n+1} = (1-lam)u_j^n + lam u_{j-1}^n
    coeffs = {0: 1-lam, -1: lam}
    M0 = sum(v for v in coeffs.values())
    M1 = sum(k*v for k, v in coeffs.items())
    M2 = sum(k**2 * v for k, v in coeffs.items())
    print('Upwind moments:')
    print('M0 =', sp.simplify(M0))
    print('M1 =', sp.simplify(M1))
    print('M2 =', sp.simplify(M2))
    print('First-order conditions satisfied:', sp.simplify(M0-1)==0 and sp.simplify(M1+lam)==0)


def verify_lax_friedrichs():
    # Lax-Friedrichs:
    # u_j^{n+1} = (1-lam)/2 u_{j+1}^n + (1+lam)/2 u_{j-1}^n
    coeffs = {1: (1-lam)/2, -1: (1+lam)/2}
    M0 = sum(v for v in coeffs.values())
    M1 = sum(k*v for k, v in coeffs.items())
    M2 = sum(k**2 * v for k, v in coeffs.items())
    print('\nLax-Friedrichs moments:')
    print('M0 =', sp.simplify(M0))
    print('M1 =', sp.simplify(M1))
    print('M2 =', sp.simplify(M2))
    print('First-order conditions satisfied:', sp.simplify(M0-1)==0 and sp.simplify(M1+lam)==0)


if __name__ == '__main__':
    verify_upwind()
    verify_lax_friedrichs()
