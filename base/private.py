import os
import re
import env
from stat import *

class Private:
    """
         Re-implementation of the bash private- for ease of use from python
 
             from env.base.private import Private
             v = Private()('DATABASE_NAME')  


         sudo -u apache python -c "from env.base.private import Private ; p=Private() ; print p('DATABASE_NAME') " 


    """
    decl = re.compile("local \s*(?P<var>\S*)=(?P<val>\S*)")
    
    def path_(self):
        epp = os.environ.get('ENV_PRIVATE_PATH',None)
        if epp: return epp  
        return os.path.join( os.path.dirname(env.HOME) , ".bash_private" )
    def __init__(self, path=None):
        if not(path):path=self.path_() 
        assert os.path.exists(path), "path %s does not exist " % path
        s = os.stat(path)
        assert S_IMODE( s.st_mode ) == S_IRUSR | S_IWUSR , "path %s has incorrect permissions " % path  
        self.path = path

    def __call__(self, qwn):
        val = ""
        f = file(self.path, "r")
        for line in f.readlines():    
            line = line.strip()
            m = Private.decl.match(line)
            if m:
                d = m.groupdict()
                if d['var'] == qwn:val = d['val']  
        f.close()
        return val
    def __repr__(self):return "<Private %s>" % self.path

if __name__=='__main__':
    import sys
    for v in sys.argv[1:]:
        print Private()(v)

