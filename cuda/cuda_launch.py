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
import os, logging, argparse
log = logging.getLogger(__name__)
from operator import mul
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
    main()



