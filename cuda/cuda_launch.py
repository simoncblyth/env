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


"""
import argparse
from operator import mul
from collections import OrderedDict

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



class Launch(object):
    def __init__(self, worksize, max_blocks=1024, threads_per_block=64 ):
        self.worksize = worksize
        self.max_blocks = max_blocks
        self.threads_per_block = threads_per_block
        pass

    chunker = property(lambda self:chunk_iterator(self.worksize, self.threads_per_block, self.max_blocks))
    counts = property(lambda self:[_[1] for _ in self.chunker])
    block = property(lambda self:(self.threads_per_block,1,1))

    def __str__(self):
        def present_launch((offset, count, blocks_per_grid)):
            grid=(blocks_per_grid, 1)
            block=(self.threads_per_block,1,1)
            return "offset %10s count %s grid %s block %s " % ( offset, count, repr(grid), repr(block) )
        pass
        return "\n".join([self.smry]+map(present_launch, self.chunker))

    def _get_smry(self):
        counts = self.counts
        assert sum(counts) == self.worksize
        return "%s worksize %s max_blocks %s threads_per_block %s launches %s " % (self.__class__.__name__, self.worksize, self.max_blocks, self.threads_per_block, len(counts))
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

    worksize = property(lambda self:reduce(mul,self.args.worksize,1))

    def _settings(self, args, defaults):
        wid = 20
        fmt = " %-15s : %20s : %20s "
        return "\n".join([ fmt % (k,str(v)[:wid],str(getattr(args,k))[:wid]) for k,v in defaults.items() ])

    def __repr__(self):
        return self._settings( self.args, self.defaults )
 

def main():
    config = Config(__doc__)
    print config.args.worksize, " => ", config.worksize
    cl = Launch(config.worksize, max_blocks=config.args.max_blocks, threads_per_block=config.args.threads_per_block)
    print cl


if __name__ == '__main__':
    main()



