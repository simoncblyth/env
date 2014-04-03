#!/usr/bin/env python
"""
http://stackoverflow.com/questions/21130121/pycuda-precision-of-matrix-multiplication-code
"""
import numpy as np
import time
# import pycuda stuff
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

BLOCK_SIZE = 16

n = 4
ni = np.int32(n)

# matrix A 
a = np.random.randn(n, n)*100
a = a.astype(np.float32)

# matrix B
b = np.random.randn(n, n)*100
b = b.astype(np.float32)

# matrix B
c = np.empty([n, n])
c = c.astype(np.float32)

# allocate memory on device
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# copy matrix to memory
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# compile kernel
mod = SourceModule(open("matmul.cu", "r").read())

# get function
matmul = mod.get_function("matmul");


# set grid size
if n%BLOCK_SIZE != 0:
    grid=(n/BLOCK_SIZE+1,n/BLOCK_SIZE+1,1)
else:
    grid=(n/BLOCK_SIZE,n/BLOCK_SIZE,1)

# call gpu function
start = time.time()
matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=grid);
end = time.time()
print "Time: %.5f s"%(end-start)

# copy back the result
cuda.memcpy_dtoh(c, c_gpu)

print np.linalg.norm(c - np.dot(a,b))
print c
print np.dot(a,b)
print c - np.dot(a,b)
