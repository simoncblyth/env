#!/usr/bin/env python
"""

"""
import os, sys, logging
log = logging.getLogger(__name__)
from env.db.simtab import Table
from shapecnf import parse_args

class ShapeDB(Table):
    default_path = os.path.join(os.path.dirname(__file__),"g4_00.db")
    def __init__(self, path=None, tn=None ):
        if path is None:
            path = self.default_path
        path = os.path.abspath(path)
        Table.__init__(self, path, tn )

    def qids(self, sql ): 
        return map(lambda _:int(_[0]), self(sql))

    def around_query(self, xyzd, fields="sid"):
        vals = map(float, xyzd.split(","))
        if len(vals) == 4:
            x,y,z,d = vals
            dx,dy,dz = d, d, d
        elif len(vals) == 6:
            x,y,z,dx,dy,dz = vals
        else:
            assert 0, "unsupported about parameters, expecting either 4 or 6 comma delimited floats "
        pass
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

        if len(ids) < 1000:
            for _ in self(sql):
                print fmt % _
        else:
            log.info("too many ids to dump %s , restrict selection and try again " % len(ids) )


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


    def handle_input(self, opts, args):
        if len(args)>0:
            ids = sorted(map(int,args))
            log.info("Operate on %s shapes, selected by args : %s " % ( len(ids), ids) )
        else:
            if opts.around:
                ids = self.around( opts.around )
                log.info("Operate on %s shapes, selected by opts.around query \"%s\"  " % (len(ids),opts.around) )
            elif opts.query:
                ids = self.qids(opts.query)
                log.info("Operate on %s shapes, selected by opts.query \"%s\" " % (len(ids),opts.query) )
            else:
                ids = None

        if opts.center:
            xyz = self.centroid(ids)
            log.info("opts.center selected, will translate all %s shapes such that centroid of all is at origin, original coordinate centroid at %s " % (len(ids), xyz))
            opts.center_xyz = xyz 
        else:
            opts.center_xyz = None

        return ids



def main():
    opts, args = parse_args(__doc__)
    db = ShapeDB()
    ids = db.handle_input( opts, args )
    db.dump(ids)

if __name__ == '__main__':
    main()




