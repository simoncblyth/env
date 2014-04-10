#!/usr/bin/env python
"""
cuda_launch_2d.py
==================

"""
import numpy as np
from operator import mul

mul_ = lambda _:reduce(mul, _)          # product of elements 
div_ = lambda num,den:(num+den-1)//den  # integer division trick, rounding up without iffing around
rep_ = lambda _:"%10s [%s] " % ( str(_),mul_(_))


def launch_iterator_2d(work_dim_=(1024,768,1), launch_dim_=(2,2,1), block_dim_=(8,8,1)):
    """
    Launch constraints:
 
    #. block of threads: max dimensions (1024, 1024, 64), BUT max threads per block 1024 eg (32,32,1)
    #. launch runtime limit is the kicker, as a result split into multiple launches controlled by launch_dim

    #. grid of blocks, max block dimensions (2147483647, 65535, 65535) : hit runtime limit long before hitting these 
    #. also use multiples of 32 for block of threads dimensions to work better with the hardware

    deviceQuery::

        Device 0: "GeForce GT 750M"
          CUDA Driver Version / Runtime Version          5.5 / 5.5
          CUDA Capability Major/Minor version number:    3.0
          Total amount of global memory:                 2048 MBytes (2147024896 bytes)
          ( 2) Multiprocessors, (192) CUDA Cores/MP:     384 CUDA Cores
          ...
          Total amount of constant memory:               65536 bytes
          Total amount of shared memory per block:       49152 bytes
          Total number of registers available per block: 65536
          Warp size:                                     32
          Maximum number of threads per multiprocessor:  2048
          Maximum number of threads per block:           1024
          Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
          Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
          Maximum memory pitch:                          2147483647 bytes
          ...
          Run time limit on kernels:                     Yes
          ...
          Support host page-locked memory mapping:       Yes


    #. numpy array division is elementwise, and integer division is used when inputs are dtype int
       (unlike with python the integer division "//" are not necessary, merely indicative ) 

    """
    work_dim, launch_dim, block_dim = map(lambda _:np.array(_,dtype=int), (work_dim_, launch_dim_, block_dim_))
    if len(work_dim) == 2:
        work_dim = np.append(work_dim, 1)

    work_per_launch = reduce(div_, (work_dim,launch_dim) )

    grid_dim = reduce(div_, (work_dim,launch_dim,block_dim) )   # work_dim//launch_dim//block_dim  taking care of roundup


    launch_index = 0
    for launchIdx_y in range(launch_dim[1]):
        launch_offset_y = work_per_launch[1]*launchIdx_y
        for launchIdx_x in range(launch_dim[0]):
            launch_offset_x = work_per_launch[0]*launchIdx_x
            launch_offset = (launch_offset_x, launch_offset_y )
            launch_work = mul_(grid_dim)*mul_(block_dim)
            yield launch_index, launch_work, launch_offset, tuple(grid_dim.tolist()), tuple(block_dim.tolist()) 
            launch_index += 1



class Launch2D(object):
    def __init__(self, work=(1024,768,1), launch=(2,3,1), block=(16,16,1) ):
        """
        """
        self.work = work
        self.launch = launch
        self.block = block
        pass

    def resize(self, size):
        self.work = size

    total = property(lambda self:reduce(mul,self.work,1))
    iterator_2d = property(lambda self:launch_iterator_2d(self.work, self.launch, self.block))
    counts = property(lambda self:[_[1] for _ in self.iterator_2d])

    def _present(self):
        def present_launch( (launch_index, work_count, offset, grid, block)):
            return "launch_index %s work_count %s offset %-20s " % ( launch_index, work_count, offset,  ) + "    ".join(["grid",rep_(grid), "block",rep_(block)])
        pass
        return map(present_launch, self.iterator_2d)
    present = property(_present) 


    def annotate(self, anno=[]):
        """
        :param anno: list of per-launch annotation information, eg CUDA_PROFILE results or pycuda timings
        """
        present = self.present
        if len(anno) == 0: 
            return "\n".join([self.smry]+ present )
        elif len(present) == len(anno):
            return "\n".join([self.smry] +["%s : %s" % (l,a) for l, a in  zip(present, anno)])
        else:
            return "mismatch between anno length and present length"
       
    def __str__(self):
        return "\n".join(self.present)

    def check_counts(self):
        counts = self.counts
        assert sum(counts) == self.total

    def _get_smry(self):
        return " ".join(["%s %s" % _ for _ in zip("work launch block".split(),map(rep_,(self.work,self.launch,self.block)))])
    smry = property(_get_smry)

    def __repr__(self):
        return self.smry


if __name__ == '__main__':

    work_a = (1024,768)
    work_b = (1440,852)
    work_c = (300,200)

    work_list = [work_a,work_b,work_c] 
    #work_list = np.random.randint(300,2000,(100,2))

    for work in work_list:
        launch = Launch2D( work, launch=(2,3,1), block=(16,16,1))
        print "\n",repr(launch), "\n", launch



