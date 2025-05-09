#!/usr/bin/env python

import logging, os
log = logging.getLogger(__name__)
import argparse
from collections import OrderedDict
import numpy as np

ivec_=lambda _:map(int,_.split(","))


class View(object):
    registry = {}
    def __init__(self):
        self.registry[self.__class__.__name__] = self 
    @classmethod
    def get(cls, name):
        return cls.registry.get(name, None)


class A(View):
    eye = "-32708.375,-818298.375,-7350.,1."
    pixel2world = """
[[      2.322        0.         -70.7107  -32257.7605]
 [     -2.322        0.         -70.7107 -814506.3488]
 [      0.           3.2839       0.       -8747.281 ]
 [      0.           0.           0.           1.    ]]
        """
a = A()

class B(View):
    eye = "-32708.375,-818298.375,-7350.,1."  
    pixel2world = """
[[      3.3769       0.         -70.7107  -33352.0169]
 [     -3.3769       0.         -70.7107 -814082.7292]
 [      0.           4.7756       0.       -9382.0125]
 [      0.           0.           0.           1.    ]]
        """
b = B()

class C(View):
    """
    daeview -t 7153 --eye=-2,-2,0 --look=0,0,0 --up=0,0,1 
    """
    eye = "-19898.3184,-804471.2402,-2612.2,1."    
    pixel2world = """
[[      2.322        0.         -70.7107  -19447.7039]
 [     -2.322        0.         -70.7107 -800679.214 ]
 [      0.           3.2839       0.       -4009.481 ]
 [      0.           0.           0.           1.    ]]
       """
c = C()


class Config(object):
    def __init__(self, doc):
        parser, defaults = self._make_parser(doc)
        self.parser = parser
        self.defaults = defaults
        self.parse()

    def parse(self):
        args = self.parser.parse_args()
        logging.basicConfig(level=getattr(logging, args.loglevel), format=args.logformat )
        np.set_printoptions(precision=4, suppress=True)
        self.args = args


    def _make_parser(self, doc):
        parser = argparse.ArgumentParser(doc)

        defaults = OrderedDict()

        defaults['loglevel'] = "INFO"
        defaults['logformat'] = "%(asctime)-15s %(name)-20s:%(lineno)-3d %(message)s"

        parser.add_argument( "--loglevel",help="INFO/DEBUG/WARN/..   %(default)s")  
        parser.add_argument( "--logformat",help="%(default)s")  


        defaults['geometry'] = os.environ.get('DAE_NAME',None)
        defaults['nodes'] = "0:"
        defaults['size'] = "1024,768"
        defaults['view'] = "A"

        parser.add_argument( "--geometry", help="Path to geometry file", type=str  )
        parser.add_argument( "--nodes", help="Specification of geometry nodes", type=str  )
        parser.add_argument( "--size", help="", type=str  )
        parser.add_argument( "--view", help="", type=str  )

        defaults['threads_per_block'] = 64
        defaults['max_blocks'] = 1024     # larger max_blocks reduces the number of separate launches, and increasing launch time (BEWARE TIMEOUT)
        defaults['block'] = "16,16,1"
        defaults['launch'] = "3,2,1"
        defaults['max_time'] = 2 ; MAX_TIME_WARN = "(greater than 4 seconds leads to GPU PANIC, GUI FREEZE AND SYSTEM CRASH) "

        parser.add_argument( "--threads-per-block", help="", type=int  )
        parser.add_argument( "--max-blocks", help="", type=int  )
        parser.add_argument( "--block", help="String 3-tuple dimensions of the block of CUDA threads, eg \"32,32,1\" \"16,16,1\" \"8,8,1\" ", type=str  )
        parser.add_argument( "--launch", help="String 3-tuple dimensions of the sequence of CUDA kernel launches, eg \"1,1,1\",  \"2,2,1\", \"2,3,1\" ", type=str  )
        parser.add_argument( "--max-time", help="Maximum time in seconds for kernel launch, if exceeded subsequent launches are ABORTed " + MAX_TIME_WARN , type=float )


        defaults['max_alpha_depth'] = 10
        defaults['alpha_depth'] = 10
        defaults['kernel'] = "render_pbo"
        defaults['kernel_flags'] = "0,0"
        defaults['metric'] = None
        defaults['allsync'] = True
        defaults['cuda_profile'] = True

        parser.add_argument( "--max-alpha-depth", help="", type=int  )
        parser.add_argument( "--alpha-depth", help="", type=int  )
        parser.add_argument( "--kernel", help="", type=str  )
        parser.add_argument( "--kernel-flags", help="g_flags constant provided to kernel, used for thread time presentation eg try 20,0  ", type=str  )
        parser.add_argument( "--metric", help="One of time/node/intersect/tri or default None", type=str  )
        parser.add_argument( "--allsync", help="Sync after every launch, to catch errors earlier.", action="store_true"  )
        parser.add_argument( "--cuda-profile", help="", action="store_true"  )


        parser.set_defaults(**defaults)
        return parser, defaults

    size=property(lambda self:ivec_(self.args.size))
    block=property(lambda self:ivec_(self.args.block))
    launch=property(lambda self:ivec_(self.args.launch))

    kernel_flags=property(lambda self:ivec_(self.args.kernel_flags))

    def _get_view(self):
        return View.get(self.args.view)
    view = property(_get_view)       

    def _get_pixel2world(self):
        s = self.view.pixel2world
        s = s.lstrip().rstrip().replace("[["," ").replace("]]"," ").replace("]"," ").replace("["," ").replace("\n"," ")
        a = np.fromstring(s, sep=" ")
        assert len(a) == 16 , (len(a), a )
        return a.reshape((4,4))
    pixel2world = property(_get_pixel2world)       
    eye = property(lambda self:np.fromstring(self.view.eye,sep=","))

    def _settings(self, args, defaults, all=False):
        if args is None:return "PARSE ERROR"
        if all:
            filter_ = lambda kv:True
        else:
            filter_ = lambda kv:kv[1] != getattr(args,kv[0]) 
        pass
        wid = 20
        fmt = " %-15s : %20s : %20s "
        return "\n".join([ fmt % (k,str(v)[:wid],str(getattr(args,k))[:wid]) for k,v in filter(filter_,defaults.items()) ])

    def settings(self, all_=False):
        return self._settings( self.args, self.defaults, all_ )

    def all_settings(self):
        return "\n".join(filter(None,[
                      self.settings(True) ,
                         ]))
    def changed_settings(self):
        return "\n".join(filter(None,[
                      self.settings(False) ,
                         ]))
    def __repr__(self):
        return self.all_settings() 
 

if __name__ == '__main__':
    pass
    config = Config(__doc__)
    print config


