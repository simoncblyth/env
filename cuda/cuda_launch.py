#!/usr/bin/env python
"""
cuda_launch.py
================

Simple Tool to plan CUDA launch sequence

::

    (chroma_env)delta:render_pbo blyth$ python cuda_launch.py 1024 768 --threads-per-block 256 --max-blocks 512
    [1024, 768]  =>  786432
    Launch worksize 786432 max_blocks 512 threads_per_block 256 launches 6 
    offset          0 count 131072 grid (512, 1) block (256, 1, 1) 
    offset     131072 count 131072 grid (512, 1) block (256, 1, 1) 
    offset     262144 count 131072 grid (512, 1) block (256, 1, 1) 
    offset     393216 count 131072 grid (512, 1) block (256, 1, 1) 
    offset     524288 count 131072 grid (512, 1) block (256, 1, 1) 
    offset     655360 count 131072 grid (512, 1) block (256, 1, 1) 


CUDA Launch config basics
--------------------------

* http://cs.nyu.edu/courses/spring12/CSCI-GA.3033-012/lecture5.pdf
* http://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim


* grid of blocks (blocks per grid limits are very high, BUT see below practical limits) 

  * grid dimension gridDim.x/y/z 
  * within each block (ie for all the threads in the block), must calc gridIdx.x/y/z to identify the block  

* block of threads (threads per block limited to 1024=32*32)

  * block dimension blockDim.x/y/z 
  * within each thread, threadIdx.x/y/z identify the thread 

1D::

    int threadID = blockIdx.x * blockDim.x + threadIdx.x  // unique within block 


extend the model upwards
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* sequence of launch grids

  * launchDim.x/y/z
  * within each launch sequence, launchIdx.x/y/z identifies the launch and from launchDim.x/y/z can determine appropriate offset  

* rather than specifying a max-blocks to stay within launch timeouts, instead specify a launchDim

  * why not max-blocks, its unclear how to structure 2D work otherwise 
  


pycuda realisation
~~~~~~~~~~~~~~~~~~~

::

     sequence=(grids_per_sequence, 1)
     grid=(blocks_per_grid, 1)
     block=(threads_per_block,1,1)


2D grid of 2D blocks
~~~~~~~~~~~~~~~~~~~~~~

* http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained

::

    2D grid of 2D blocks  
     __device__ int getGlobalIdx_2D_2D()
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
        int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
        return threadId;
    }



1D launch
~~~~~~~~~~

::

     0000001111111122222222333333333444444444



2D launch
~~~~~~~~~~~

* :google:`mapping cuda thread index to pixel`
* http://stackoverflow.com/questions/9099749/using-cuda-to-find-the-pixel-wise-average-value-of-a-bunch-of-images
* https://devtalk.nvidia.com/default/topic/400713/simple-question-about-blockidx-griddim/


::

    work        (width, height )  1024x768   = 786432

    block_dim  (16,16,1) = 256             


    NOPE TOO MANY BLOCK WITH   grid_dim    (64,48)  = 3072      [ 1024/16 x 768/16 ]   [ 3072*256 = 786432 ] 3072 
       

    INTRODUCE LAUNCH DIM  2 x 2  
      => 4 launches each with  grid_dim  (32,24)  62/2 x 48/2 = 32 x 24 = 768      

                (32,24)  (32,24)
                (32,24)  (32,24)


    Will this indexing suffice ?


                     0:511                                  
            ......................................
             0:31            16          0:15
    int x = blockIdx.x * blockDim.x + threadIdx.x     + launch_offset_x ;

            gridDim.x = 32                                0:1            32          16     
                                                        launchIdx.x * gridDim.x * blockDim.x    =    0/512 

                                                        launchDim.x = 2

             
             0:23            16          0:15                0/384
    int y = blockIdx.y * blockDim.y + threadIdx.y     + launch_offset_y ; 
                          
                                                           0:1           24         16    
            gridDim.y = 24                              launchIdx.y * gridDim.y * blockDim.y    =   0/384  

                                                        launchDim.y = 2

         launchDim = 2,2
         launchIdx.x = 0:1
         launchIdx.y = 0:1


    What about 1440,852    need to provide offsets for the straggler pixels
          /16    90,53

        In [144]: np.array([1440,852])/np.array([16,16])
        Out[144]: array([90, 53])

        In [148]: np.prod(a)
        Out[148]: 4770             


   
limits
~~~~~~~

From deviceQuery::

    Device 0: "GeForce GT 750M"
      CUDA Driver Version / Runtime Version          5.5 / 5.5
      CUDA Capability Major/Minor version number:    3.0
      ...
      Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
      Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
      Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
      Total amount of constant memory:               65536 bytes
      Total amount of shared memory per block:       49152 bytes
      Total number of registers available per block: 65536
      Warp size:                                     32
      Maximum number of threads per multiprocessor:  2048
      Maximum number of threads per block:           1024
      Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
      Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
      ...

practical limits
~~~~~~~~~~~~~~~~

On OSX, need to restrict kernel launch times to less than 5 seconds, otherwise they get 
killed, GPU panics and hard system crashes result. Upshot is must restrict the 
number of blocks within the grid and split expensive processing into multiple launches 
to keep each launch within the timeout.
 

"""
import os, logging, argparse
log = logging.getLogger(__name__)
from operator import mul
mul_ = lambda _:reduce(mul, _)

from collections import OrderedDict

from env.cuda.cuda_profile_parse import Parser


def chunk_iterator(nelements, nthreads_per_block=64, max_blocks=1024):
    """
    Extracted from chroma.gpu.tools  

    Iterator that yields tuples with the values requried to process
    a long array in multiple kernel passes on the GPU.

    Each yielded value is of the form,
        (first_index, elements_this_iteration, nblocks_this_iteration)

    Example:
        >>> list(chunk_iterator(300, 32, 2))
        [(0, 64, 2), (64, 64, 2), (128, 64, 2), (192, 64, 2), (256, 9, 1)]
    """
    first = 0 
    while first < nelements:
        elements_left = nelements - first
        blocks = int(elements_left // nthreads_per_block)
        if elements_left % nthreads_per_block != 0:
            blocks += 1 # Round up only if needed
        blocks = min(max_blocks, blocks)
        elements_this_round = min(elements_left, blocks * nthreads_per_block)

        yield (first, elements_this_round, blocks)
        first += elements_this_round


def launch_iterator_1d( work, block_dim=(64,1,1), max_blocks_per_grid=1024):
    work = mul_(work)   # express as tuple/list even when 1d please

    threads_per_block = mul_(block_dim)

    offset = 0
    launch = 0
    while offset < work:
        left = work - offset
        #
        blocks_per_grid = int( left // threads_per_block )
        #
        if not left % threads_per_block == 0:
            blocks_per_grid += 1 

        blocks_per_grid = min(max_blocks_per_grid, blocks_per_grid)
        grid = (blocks_per_grid, 1)
        #
        done_this_launch = min(left, mul_(grid)*threads_per_block )
 
        yield launch, offset, grid, block_dim

        offset += done_this_launch
        launch += 1


def launch_iterator_2d( work, block_dim=(8,8,1), launch_dim=(2,2)):
    """

    """
    pass






class CUDACheck(object):
    log = "cuda_profile_0.log"
    def __init__(self, config):
        if config.args.cuda_profile:
            log.info("setting CUDA_PROFILE envvar, will write logs such as %s " % self.log )
            os.environ['CUDA_PROFILE'] = "1"
        self.config = config
        self.kernel = config.args.kernel
        self.parser = Parser()
        self.profile = []

    def parse_profile(self):
        self.profile = []
        if self.config.args.cuda_profile:
            for d in self.parser(self.log):
                #print d
                if d['method'].startswith(self.kernel):
                    self.profile.append(d)
    
    def compare_with_launch_times(self, times, launch):
        nprofile = len(self.profile)
        nlaunch = len(times)
        log.info("nprofile %s nlaunch %s" % (nprofile, nlaunch))
        if nprofile > nlaunch:
            profile = self.profile[-nlaunch:]
        elif nprofile == nlaunch:
            profile = self.profile
        else:
            log.info("times : %s " % repr(times))
            log.warn("unexpected")
            return  

        assert len(profile) == nlaunch
        anno = ["%15.5f" % t + " %(gputime)15.1f %(cputime)15.3fs %(occupancy)s " % d for d,t in zip(profile, times)]
        print launch.present(anno)




class Launch(object):
    def __init__(self, size, max_blocks=1024, threads_per_block=64 ):
        """
        :param size: 1/2/3-dimensional tuple/array with work size eg (1024,768) 
        """
        self.size = size
        self.max_blocks = max_blocks
        self.threads_per_block = threads_per_block
        pass

    def resize(self, size):
        self.size = size

    total = property(lambda self:reduce(mul,self.size,1))
    chunker = property(lambda self:chunk_iterator(self.total, self.threads_per_block, self.max_blocks))
    counts = property(lambda self:[_[1] for _ in self.chunker])
    block = property(lambda self:(self.threads_per_block,1,1))

    def present(self, anno=[]):
        def present_launch((offset, count, blocks_per_grid)):
            grid=(blocks_per_grid, 1)
            block=(self.threads_per_block,1,1)
            return "offset %10s count %s grid %s block %s " % ( offset, count, repr(grid), repr(block) )
        pass

        launches = map(present_launch, self.chunker)
        if len(anno) == 0: 
            return "\n".join([self.smry]+ launches )
        elif len(launches) == len(anno):
            return "\n".join([self.smry] +["%s : %s" % (l,a) for l, a in  zip(launches, anno)])
        else:
            return "mismatch between anno length and launch length"
       

    def __str__(self):
        return self.present()
    def _get_smry(self):
        counts = self.counts
        assert sum(counts) == self.total
        return "%s size %s total %s max_blocks %s threads_per_block %s launches %s block %s " % (self.__class__.__name__, self.size, self.total, \
                     self.max_blocks, self.threads_per_block, len(counts), repr(self.block) )
    smry = property(_get_smry)

    def __repr__(self):
        return self.smry



class Config(object):
    def __init__(self, doc):
        parser, defaults = self._make_parser(doc)
        self.defaults = defaults
        self.args = parser.parse_args()
 
    def _make_parser(self, doc):
        parser = argparse.ArgumentParser(doc)

        defaults = OrderedDict()
        defaults['threads_per_block'] = 64
        defaults['max_blocks'] = 1024     # larger max_blocks reduces the number of separate launches, and increasing launch time (BEWARE TIMEOUT)

        parser.add_argument( "worksize", nargs='+',  help="One or more integers the product of which is the total worksize", type=int  )
        parser.add_argument( "-t","--threads-per-block", help="", type=int  )
        parser.add_argument( "-b","--max-blocks", help="", type=int  )

        parser.set_defaults(**defaults)
        return parser, defaults


    def _settings(self, args, defaults):
        wid = 20
        fmt = " %-15s : %20s : %20s "
        return "\n".join([ fmt % (k,str(v)[:wid],str(getattr(args,k))[:wid]) for k,v in defaults.items() ])

    def __repr__(self):
        return self._settings( self.args, self.defaults )
 

def main():
    config = Config(__doc__)
    cl = Launch(config.args.worksize, max_blocks=config.args.max_blocks, threads_per_block=config.args.threads_per_block)
    print cl


if __name__ == '__main__':
    #main()

    for launch, offset, grid, block in launch_iterator_1d( (1024,768,) ):
        print "launch %s offset %s grid %s block %s " % ( launch, offset, repr(grid), repr(block))     
        




