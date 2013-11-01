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

    def _extract(self, tok):
        s = self.src
        beg = s.index(tok)
        end = s.index("]", beg)
        strip_ = lambda _:_.lstrip().rstrip()
        return filter(None,map(strip_,s[beg+len(tok):end-1].replace(","," ").split("\n")))
        
    def _points(self):
        lines = self._extract(tok="point [")
        return numpy.fromstring(" ".join(lines), dtype=numpy.float32, sep=' ').reshape(len(lines), 3 )

    def _faces(self):
        arr = []
        for line in self._extract(tok="coordIndex ["):
            assert line[-3:] == ' -1', line
            line = line[:-3].rstrip().replace("  "," ") 
            arr.append(map(int,line.split(" ")))
        return arr

    points = property(_points)
    faces = property(_faces)    


def main():
    db = VRML2DB()
    sh = db.shape(sys.argv[1])
    print "*" * 100, "\n",sh.src
    print "*" * 100, "\n",sh.points
    print "*" * 100
    for _ in sh.faces:print _


if __name__ == '__main__':
    main()

