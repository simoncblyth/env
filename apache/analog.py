import os, sys
import numpy as np

"""
Usage::

   from env.apache.analog import load
   a = load("/tmp/env/analog/access_log/0_1000")

"""

def load( path , chk_orig=True ):
    """
       avoid trivial numpy bug 
            global name 'fh' is not defined

       by passing in a fh 
    """
    print "load from %s   " % ( path ) 
    own_fh = False
    if not hasattr(path,'read'):
        own_fh = True
        fh = open(path, "r")

    
    fields = [ ('ip','S15'), ('id', 'S10') , ('user', 'S10'), ('time','M8[s]'), ('req','S100'), ('resp','i4'), ('size','i4') ] 
    patn = r'(\d+\.\d+\.\d+\.\d+) (.+) (.+) \[(.*)\] "(.*)" (\d+) ([\d-]+)'

    if chk_orig:
        t = np.dtype([ ('orig','S256') ] + fields )   
        r = r'(%s)' % patn
    else:
        t = np.dtype( fields )
        r = patn 

    a = np.fromregex( fh if own_fh else path , r, t )

    if own_fh:
        fh.close()

    if chk_orig:
        name = path + '.chk_orig' 
        o = open(name, "w")
        print "writing  %s " % name 
        o.write( "\n".join( a['orig'] ) + "\n" )
        o.close()  

    return a 




def array_save( path ):
    """
         failing with 
           ValueError: mismatch in size of old and new data-descriptor
    """
    z = path + '.npz'
    print "array_save from %s into %s  " % (path, z) 
    a = load(path)
    print "array_save into %s length %s " % ( z, len(a)  ) 
    n = os.path.basename( path )
    np.savez( z , a=a )

if __name__ == '__main__':
    assert len(sys.argv) == 2
    array_save( sys.argv[1] )



