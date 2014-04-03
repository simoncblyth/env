#!/usr/bin/env python
"""
http://stackoverflow.com/questions/21130121/pycuda-precision-of-matrix-multiplication-code


Curious, the structuring degrades the agreement a bit ? 

"""
import numpy as np
import time

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


class Matmul(object):
    def __init__(self):
        self.init_gpu() 

    def init_gpu(self):
        matmul = SourceModule(open("matmul.cu", "r").read()).get_function("matmul")
        self.matmul = matmul

    def __call__(self, n, A, B, BLOCK_SIZE=16):
        C = np.empty([n, n])
        n = np.int32(n) 

        self.a = A.astype(np.float32)
        self.b = B.astype(np.float32)
        self.c = C.astype(np.float32)

        start = time.time()
        self.launch( n, BLOCK_SIZE )
        end = time.time()
        print "Time: %.5f s"%(end-start)

        return self.c

    def launch(self, n, BLOCK_SIZE ):
        block = (BLOCK_SIZE,BLOCK_SIZE,1)
        grid =  (n/BLOCK_SIZE,n/BLOCK_SIZE,1) if n % BLOCK_SIZE == 0 else (n/BLOCK_SIZE+1,n/BLOCK_SIZE+1,1) 

        a_gpu = cuda.mem_alloc(self.a.nbytes)
        b_gpu = cuda.mem_alloc(self.b.nbytes)
        c_gpu = cuda.mem_alloc(self.c.nbytes)

        cuda.memcpy_htod(a_gpu, self.a)
        cuda.memcpy_htod(b_gpu, self.b)

        self.matmul(n, a_gpu, b_gpu, c_gpu, block=block, grid=grid)

        cuda.memcpy_dtoh(self.c, c_gpu)



if __name__ == '__main__':

    np.set_printoptions(precision=5, suppress=True)

    n = 4
    A = np.random.randn(n, n)*100
    B = np.random.randn(n, n)*100

    MM = Matmul()
    C = MM(n,A,B)
    CNP = np.dot(A,B) 

    print np.linalg.norm(C - CNP)
    print C
    print CNP
    print C - CNP
