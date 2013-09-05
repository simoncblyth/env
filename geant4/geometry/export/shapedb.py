#!/usr/bin/env python
"""

"""
import os, sys, logging
log = logging.getLogger(__name__)
from env.db.simtab import Table

class ShapeDB(Table):
    default_path = os.path.join(os.path.dirname(__file__),"g4_00.db")
    def __init__(self, path=None, tn=None ):
        if path is None:
            path = self.default_path
        path = os.path.abspath(path)
        Table.__init__(self, path, tn )

    def qids(self, sql ): 
        """
        :param sql: 
        """
        return map(lambda _:int(_[0]), self(sql))

    def around_query(self, xyzd, fields="sid"):
        x,y,z,d = map(float, xyzd.split(","))
        print x,y,z,d
        dx,dy,dz = d, d, d
        return "select %(fields)s from xshape where abs(ax-(%(x)s)) < %(dx)s and abs(ay-(%(y)s)) < %(dy)s and abs(az-(%(z)s)) < %(dz)s ;" % locals()

    def around(self, xyzd):
        sql = self.around_query(xyzd)
        return self.qids(sql)

    def dump_query(self, ids, fields ):
        sids = ",".join(map(str,ids))
        return "select %(fields)s from xshape where sid in (%(sids)s) ;" % locals()

    def dump(self, ids, xfields="ax,ay,az,dx,dy,dz" ):
        xfields = xfields.split(",")
        fmt = " %10d " + " %10.2f " * len(xfields)
        lfmt = " %10s " + " %10s " * len(xfields)
        fields = ["sid"] + xfields
        sql = self.dump_query(ids, ",".join(fields))
        print lfmt % tuple(fields)
        for _ in self(sql):
            print fmt % _

    def centroid(self, ids):
        """
        Averaging a list of volumes
        -----------------------------

        Use a pair from the degenerate dozen to demo shapeset averaging::

            sqlite> select npo, sumx, sumy, sumz, sumx/npo, sumy/npo, sumz/npo, ax,ay,az  from xshape where sid in (6400,6401) ;
            npo         sumx        sumy         sumz        sumx/npo    sumy/npo    sumz/npo           ax          ay          az               
            ----------  ----------  -----------  ----------  ----------  ----------  -----------------  ----------  ----------  -----------------
            50          -797559.0   -40289110.0  -207856.0   -15951.18   -805782.2   -4157.12000000001  -15951.18   -805782.2   -4157.12000000001
            50          -797559.0   -40289110.0  -207856.0   -15951.18   -805782.2   -4157.12000000001  -15951.18   -805782.2   -4157.12000000001

            sqlite> select sum(sumx)/sum(npo) as ssx, sum(sumy)/sum(npo) as ssy, sum(sumz)/sum(npo) as ssz from  xshape where sid in (6400,6401) ;
            ssx         ssy         ssz              
            ----------  ----------  -----------------
            -15951.18   -805782.2   -4157.12000000001

        """
        sids = ",".join(map(str,ids))
        sql = "select sum(sumx)/sum(npo) as ssx, sum(sumy)/sum(npo) as ssy, sum(sumz)/sum(npo) as ssz from  xshape where sid in (%s) " % sids
        lret = map(lambda _:(_[0],_[1],_[2]),self(sql))
        assert len(lret) == 1 , (lret, sql)
        return lret[0]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    db = ShapeDB()
    print db.centroid([6400,6401])
    ids = db.around("-16444.75,-811537.5,-1350,100")
    print ids
    db.dump(ids)





