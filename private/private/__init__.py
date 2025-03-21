import os
import re
from stat import *
    
def priv(*args):
   p = Private()
   return p(*args) 
    
class Private(dict):
    """
         Re-implementation of the bash private- for ease of use from python
 
             from private import Private
             v = Private()('DATABASE_NAME')  


         sudo -u apache python -c "from :private import Private ; p=Private() ; print p('DATABASE_NAME') " 

    """
    decl = re.compile("local \s*(?P<var>\S*)=(?P<val>.*)$")
    def __init__(self):
        #print "\n".join(["%s:%s" % (k,v) for k,v in os.environ.items()]) 
        path = os.environ.get('ENV_PRIVATE_PATH',None)
        if(not(path)):
            path = os.path.join( os.path.dirname(os.environ.get('ENV_HOME')), '.bash_private' )
        assert path , "path not defined : this can be set with envvar ENV_PRIVATE_PATH not defined "        
        self.path = path
        assert os.path.exists(path), "path does not exist ... %s  %s " % ( path , self )
        s = os.stat(path)
        assert S_IMODE( s.st_mode ) == S_IRUSR | S_IWUSR , "incorrect permissions .. %s  " % ( self )  
        self.parse()

    def parse(self):
        f = file(self.path, "r")
        for line in f.readlines():    
            line = line.strip()
            m = Private.decl.match(line)
            if m:
                d = m.groupdict()
                self[d['var']] = d['val']
        f.close()

    def __call__(self, *args, **kwa ):
        if len(kwa) == 0:
            assert len(args) == 1 , "error expecting one arg %s " % repr(args)
            return self.get(args[0], None )
        else:  
            return dict( [ (k,self(v),) for k,v in kwa.items() ] ) 

    def __repr__(self):return "\n".join( ["<Private %s>" % self.path ] )

if __name__=='__main__':
    import sys
    for v in sys.argv[1:]:
        print Private()(v)

