#!/usr/bin/env python

import os, sys, numpy, sqlite3

class VRML2DB(object):
    def __init__(self, path="$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db"):
        db = sqlite3.connect(os.path.expandvars(path))
        self.db = db

    def __call__(self, sql):
        return list(self.db.execute(sql))[0][0]

    def shape(self, id):
        src = self("select src from shape where id=%s limit 1" % id )
        return VRML2SRC(src)

class VRML2SRC(object):
    def __init__(self, src):
        self.src = src

    def _points(self):
        s = self.src
        lines = filter(None,map(lambda _:_.lstrip(),s[s.index("point [")+7:s.index("]")-1].lstrip().rstrip().split(",")))
        return numpy.fromstring(" ".join(lines), dtype=numpy.float32, sep=' ').reshape( (len(lines),3) )
        
    points = property(_points)    
    


def main():
    db = VRML2DB()
    sh = db.shape(sys.argv[1])
    print "*" * 100, "\n",sh.src
    print "*" * 100, "\n",sh.points


if __name__ == '__main__':
    main()

