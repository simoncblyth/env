import os
import numpy as np

class Scan(dict):
    """
        define scan by the record array used to keep the results 
        this should allow multi-dimensional scans with little modification
    """
    def __init__(self, *args, **kwargs ): 
        dict.__init__(self, *args, **kwargs)
        self.update( class_=self.__class__.__name__  )
        self.cursor = -1
        self._kwargs = kwargs

    def __iter__(self):
        return self

    def next(self):
        """
             Cranking the iterator updates self to the new parameters of the scan
        """
        self.cursor += 1
        if self.cursor > len(self.scan) - 1:
            self.cursor = -1
            raise StopIteration
        rec = self.scan[self.cursor]
        self.update( dict(zip(rec.dtype.names,rec)) )
        return self 

    kwargs = property( lambda self:",".join( [ "%s=\"%s\"" % _ for _ in self._kwargs.items() ]))
    setup = property( lambda self:self._setup % self )
    stmt =  property( lambda self:self._stmt % self )
    name =  property( lambda self:self._name % self )
    path = property(lambda self:".npz/%s.npz" % self.name )
 
    def __call__(self, **kwargs ):
        """
             Stores results derived from the supplied context into the record array 
        """
        for k, v in kwargs.items():  
            if k in self.scan.dtype.names:
                self.scan[self.cursor][k] = v
      
    def __str__(self):
        return " %s %s %s " % ( repr(dict(self)), self.setup, self.stmt )  

    def __repr__(self):
        return "%s(%s)[%d] " % ( self.__class__.__name__ , self.kwargs , self.cursor )  

    def run(self):
        """
             Iterate over self, changing context according to the scan record array and
             running the timed commands
        """
        for _ in self:
            self()


    def save(self):
        p = self.path
        dir = os.path.dirname( p )
        if not os.path.exists(dir):
            os.makedirs(dir)

        keys = ['cls'] + self._kwargs.keys()
        vals = [ self.__class__.__name__ ] + self._kwargs.values() 
        descr = list( (k,'S10') for k in keys  )
        print descr
        meta = np.ndarray( (1,) , dtype=np.dtype(descr) )

        for k,v in zip(keys,vals):
            meta[0][k] = str(v) 

        print "save scan into %s " % p 
        print repr(meta)
        np.savez( p ,  scan=self.scan, meta=meta  )



