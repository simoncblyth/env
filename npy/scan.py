import os
import numpy as np

from env.memcheck.mem import Ps
_ps = Ps(os.getpid())
rss = lambda:_ps.rss_megabytes

import time
timer = time.time



class ScanIt(dict):
    """ Parameter context scan and result record array used to keep the results """
    steps = 5
    max = 10000 
    min = 1 
    def __init__(self, *args, **kwargs ): 
        dict.__init__(self, *args, **kwargs)
        it = args[0] 
        self.it = it

        self.update( scan=self.__class__.__name__  )

        self.cursor = -1
        self._kwargs = kwargs
        assert self.steps and self.max, "subclass Scan, and supply these attributes " 
        scan = np.zeros( (self.steps,) , np.dtype([('limit','i4'),('ctime','f4'),('itime','f4'),('rss_','f4')]) ) 
        scan['limit'] = np.linspace( self.min , self.max , len(scan) )
        self.scan = scan

    def __iter__(self):
        return self

    def next(self):
        """ Cranking the iterator updates self to the new parameters of the scan """
        self.cursor += 1
        if self.cursor > len(self.scan) - 1:
            self.cursor = -1
            raise StopIteration
        rec = self.scan[self.cursor]
        self.it.update( dict(zip(rec.dtype.names,rec)) )
        return self.it   ## does this do a deep copy ? 

    kwargs = property( lambda self:",".join( [ "%s=\"%s\"" % _ for _ in self._kwargs.items() ]))
    path = property(lambda self:".npz/%s.npz" % self.get('callable','NoCallable') )
 
    def __call__(self, **kwargs ):
        """ Stores results into the record array slot for current context cursor  """
        for k, v in kwargs.items():  
            if k in self.scan.dtype.names:
                self.scan[self.cursor][k] = v
            else:
                self.update({ k:v })
      
    def __repr__(self):
        return "%s(%s)[%d] " % ( self.__class__.__name__ , self.kwargs , self.cursor )  

    def _meta(self):
        keys = self.keys()
        vals = self.values()
        m = {}
        for k,v in zip(keys,vals):
            if type(v) == str:
                m[k] = str(v) 
        descr = list( (k,'S%d' % len(v)  ) for k,v in m.items()  )
        meta = np.ndarray( (1,) , dtype=np.dtype(descr) )
        for k,v in m.items():
            meta[0][k] = str(v) 
        return meta
    meta = property(_meta)

    def save(self):
        p = self.path
        dir = os.path.dirname( p )
        if not os.path.exists(dir):
            os.makedirs(dir)
        print "save scan into %s " % p 
        print repr(self.meta)
        np.savez( p ,  scan=self.scan, meta=self.meta  )


class DeprecatedOldTimr(dict):
    """
         timeit is a pain to use .. as have to squeeze everything into strings ...
         and are tied to problem under test
    """
    _name  = "%(class_)s_%(kls)s_%(method)s"
    _setup = "from tech import %(kls)s ; callable = %(kls)s()  "
    _stmt =  "callable(method=%(method)s, limit=%(limit)s) ; del callable ; callable=None " 

    setup = property( lambda self:self._setup % self )
    stmt =  property( lambda self:self._stmt % self )
    name =  property( lambda self:self._name % self )

    def __call__(self):
        timer = Timer( self.stmt, self.setup )
        try:
            time_=timer.timeit(1)  
        except:
            time_=-1
            timer.print_exc()
            sys.exit(1)

        res = dict( time_=time_, rss_=rss(), task_=self.name )
        self.update( **res ) 
        if 'verbose' in self:
            print self
        return res



class CallTimr(dict):
    def __init__(self, callable ):
        self.callable = callable

    def __call__(self, *args, **kwargs ):
        """
             Record the time to call the callable 
             and monitor memory usage after the call
        """
        t0 = timer()
        self.callable( *args, **kwargs )
        time_ = timer() - t0
        res = dict( time_=time_, rss_=rss(), task_=self.callable )
        self.update( **res ) 
        if 'verbose' in self:
            print self
        return res


class ClassTimr(dict):

    def __init__(self, class_, *args, **kwargs ):
        self.class_ = class_
        t0 = timer()
        self.callable = class_( *args, **kwargs)
        self.update( itime=timer() - t0 )

    def __call__(self, *args, **kwargs ):
        """
             Record the time to call the callable 
             and monitor memory usage after the call
        """
        callable = self.callable

        t0 = timer()
        callable( *args, **kwargs )

        self.update( ctime=timer() - t0 , rss_=rss(), callable=repr(callable), symbol=callable.symbol )

     

if __name__ == '__main__':
    pass




