#!/usr/bin/env python

import os, sys, numpy, sqlite3

class VRML2DB(object):
    def __init__(self, path="$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db"):
        db = sqlite3.connect(os.path.expandvars(path))
        self.db = db

    def points(self, id):
        s = list(self.db.execute("select src from shape where id=%s limit 1" % id))[0][0]
        lines = filter(None,map(lambda _:_.lstrip(),s[s.index("point [")+7:s.index("]")-1].lstrip().rstrip().split(",")))
        a = numpy.fromstring(" ".join(lines), dtype=numpy.float32, sep=' ').reshape( (len(lines),3) )
        return a 

if __name__ == '__main__':
    db = VRML2DB()
    print db.points(sys.argv[1])


