#!/usr/bin/env python
"""
"""

import os, sqlite3

if __name__ == '__main__':
    pass
    db = sqlite3.connect(":memory:")
    dbs =  { 
                'wrl':"$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db", 
                'dae':"$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.db",
           }
    for tag, path in dbs.items():
        path = os.path.expandvars(path)
        db.execute('attach database "%(path)s" as %(tag)s' % locals() )

    for _ in db.execute("select count(*) from dae.geom "):print _
    for _ in db.execute("select count(*) from wrl.xshape "):print _
