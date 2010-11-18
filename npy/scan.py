import os
import numpy as np

class Scan(dict):
    """ Parameter context scan and result record array used to keep the results """
    def __init__(self, *args, **kwargs ): 
        dict.__init__(self, *args, **kwargs)
        self.update( class_=self.__class__.__name__  )
        self.cursor = -1
        self._kwargs = kwargs
        assert self.steps and self.max, "subclass Scan, and supply these attributes " 
        scan = np.zeros( (self.steps,) , np.dtype([('limit','i4'),('time_','f4'),('rss_','f4')]) ) 
        scan['limit'] = np.linspace( 0, self.max , len(scan) )
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
        self.update( dict(zip(rec.dtype.names,rec)) )
        return dict(self)   ## does this do a deep copy ? 

    kwargs = property( lambda self:",".join( [ "%s=\"%s\"" % _ for _ in self._kwargs.items() ]))
    path = property(lambda self:".npz/%s.npz" % self.get('task_','NoTask') )
 
    def __call__(self, **kwargs ):
        """ Stores results into the record array slot for current context cursor  """
        self['task_'] = kwargs.pop('task_')   
        for k, v in kwargs.items():  
            if k in self.scan.dtype.names:
                self.scan[self.cursor][k] = v
      
    def __repr__(self):
        return "%s(%s)[%d] " % ( self.__class__.__name__ , self.kwargs , self.cursor )  

    def _meta(self):
        keys = ['cls'] + self._kwargs.keys()
        vals = [ self.__class__.__name__ ] + self._kwargs.values() 
        descr = list( (k,'S10') for k in keys  )
        meta = np.ndarray( (1,) , dtype=np.dtype(descr) )
        for k,v in zip(keys,vals):
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



