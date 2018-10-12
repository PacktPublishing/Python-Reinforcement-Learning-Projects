'''
Created on 4 Sep 2017

@author: ywz
'''
import numpy


class Krylov:
    
    def __init__(self):
        pass
    
    def cg(self, Ax, b, cg_iters=10, verbose=False, eps=1e-10):
        
        x = numpy.zeros_like(b)
        r = b.copy()
        p = b.copy()
        r_dot_r = r.dot(r)
        
        for _ in range(cg_iters):
            z = Ax(p)
            v = r_dot_r / p.dot(z)
            x += v * p
            r -= v * z
            
            new_r_dot_r = r.dot(r)
            beta = new_r_dot_r / r_dot_r
            p = r + beta * p
            
            r_dot_r = new_r_dot_r
            if r_dot_r < eps:
                break
        
        if verbose: 
            print("residual norm: {:5f}, solution norm: {:5f}".format(r_dot_r, numpy.linalg.norm(x)))
        return x
    
if __name__ == "__main__":
    
    from numpy.linalg import inv
    
    n = 5
    A = numpy.random.rand(n, n)
    A = A.T.dot(A) + 0.01 * numpy.eye(n)
    b = numpy.random.rand(n)
    x = inv(A).dot(b)

    krylov = Krylov()
    y = krylov.cg(lambda x: A.dot(x), b, verbose=True)
    
    print(x)
    print(y)
    
    

            