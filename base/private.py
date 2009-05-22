import os
import re

class Private:
    """
         Re-implementation of the bash private- for ease of use from python
 
             from env.base.private import Private
             v = Private()('DATABASE_NAME')  

    """
    decl = re.compile("local \s*(?P<var>\S*)=(?P<val>\S*)")
    path = "%s/%s" % ( os.environ['HOME'] , ".bash_private" )
    def __init__(self, path=path):
        assert os.path.exists(path)
        self.path = path
        ll=os.popen("ls -l %s" % self.path).read()
        assert ll.split()[0] == '-rw-------'
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

