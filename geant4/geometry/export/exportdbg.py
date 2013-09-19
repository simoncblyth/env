#!/usr/bin/env python
"""
sqlite> .w 7 7 7 7 7 7 7 7 150
sqlite> select sid,npo,ax,ay,az,dx,dy,dz,name from xshape where sid in (2435,3149,3151,3199,4356,4447,4463,4539,4540,4542,4550,4551,4565,4566,4605) ; 
sid      npo      ax       ay       az       dx       dy       dz       name                                                                                                                                                  
-------  -------  -------  -------  -------  -------  -------  -------  ---------------------------------------------------------------------------------------------                                                         
2435     24       -20103.  -796552  -1583.3  9892.5   6398.0   294.4    /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearRightSpanHbeam2.1003                                                                                 
3149     50       -16048.  -803091  -7067.9  13644.5  15422.0  9916.0   /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner.1000                                                                                                 
3151     50       -16085.  -802990  -6565.9  11506.8  13286.0  8912.0   /dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain.1000                                                                                                
3199     362      -16595.  -801442  -8842.5  270.200  286.0    200.580  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt.1                                       
4356     50       -18289.  -800004  -4826.5  60.7999  61.0     53.0     /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtBase.1001                                                                                                 
4447     16       -18423.  -797823  -9355.0  1481.6   330.0    430.0    /dd/Geometry/AD/lvOIL#pvSstBotCirRib#SstBotCirRib:7.7                                                                                                 
4463     22       -17784.  -798325  -4814.5  553.700  2147.0   258.730  /dd/Geometry/AD/lvOIL#pvSstTopTshapeRibs#SstTopTshapeRibs:7#SstTopTshapeRot.7                                                                         
4539     203      -18014.  -799605  -4226.9  525.299  525.0    12.6999  /dd/Geometry/CalibrationBox/lvDomeInterior#pvTurntableLowerPlate.1002                                                                                 
4540     629      -18079.  -799699  -4117.2  20.0999  20.0     204.61   /dd/Geometry/CalibrationBox/lvDomeInterior#pvLedSourceAssyInAcu.1003                                                                                  
4542     267      -18079.  -799699  -4157.1  15.9000  15.0     15.8800  /dd/Geometry/CalibrationSources/lvLedSourceShell#pvDiffuserBall.1000                                                                                  
4550     50       -18079.  -799699  -4194.7  0.59999  1.0      25.3999  /dd/Geometry/CalibrationSources/lvLedSourceAssy#pvLedWeightCableBot.1004                                                                              
4551     357      -17900.  -799614  -4125.3  20.0     20.0     204.61   /dd/Geometry/CalibrationBox/lvDomeInterior#pvGe68SourceAssyInAcu.1004                                                                                 
4565     50       -17900.  -799614  -4194.7  0.60000  1.0      25.3999  /dd/Geometry/CalibrationSources/lvGe68SourceAssy#pvWeightCableBot.1004                                                                                
4566     296      -18063.  -799502  -4106.4  20.0     20.0     204.61   /dd/Geometry/CalibrationBox/lvDomeInterior#pvAmCCo60SourceAssyInAcu.1005                                                                              
4605     172      -18056.  -799666  -4389.4  1845.7   1845.0   299.99   /dd/Geometry/OverflowTanks/lvOflTnkContainer#pvOflTnkCnrSpace.1000                                                                                    
sqlite> 


"""
import sys, os, logging
from env.db.simtab import Table
log = logging.getLogger(__name__)

class ShapeDB(Table):
    def __init__(self, path=None, tn=None ):
        path = os.path.abspath(path)
        log.info("opening %s " % path)
        Table.__init__(self, path, tn )

    def lookup(self, id):
        rec = self.getone("select name,id from shape where id=%(id)s" % locals())
        return rec 

    def cli(self, sql):
        cmd = "echo \"%(sql)s\" | sqlite3 %(path)s " % dict(sql=sql,path=self.path)
        print os.popen(cmd).read()  

    def dump(self, ids):
        sids = ",".join(map(str,ids))
        sql = "select sid,npo,ax,ay,az,dx,dy,dz,name from xshape where sid in (%(sids)s) ; " % locals()
        print sql 
        self.cli(sql)





class LogParser(dict):
    """
    """
    beg = r"""Traversing scene data..."""
    end = r"""Viewer "viewer-0 (VRML2FILE)" refreshed."""
    def parse(self, path):
        ipv = -1 # make Universe.0 come out as ipv 0, to be in line with the culled 
        pv, region, token = None, "head", False
        for line in file(path).readlines():
            line = line.strip()
            if line == self.beg:
                region, token = "traverse", True
            elif line == self.end:
                return
            elif line[0:4] == 'SCB ':
                token = True 
                pv = line[4:]
                ipv += 1 
                log.debug("[%s][%s]" % (region,pv))
            else:
                token = False 

            if pv and not token and region == 'traverse': 
                if not ipv in self:
                    self[ipv] = dict(ipv=ipv,pv=pv,lines=[])
                self[ipv]['lines'].append(line)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    lp = LogParser()
    lp.parse("exportdbg.log")

    db = ShapeDB("g4_01.db")

    for ipv in sorted(lp.keys()):
        pvdb = db.lookup(ipv)
        rec = lp[ipv] 
        assert rec['ipv'] == ipv
        assert pvdb == rec['pv']  
        print "\n".join(["",str(ipv),rec['pv']] + rec['lines'])
        db.dump([ipv])     

    ids = sorted(lp.keys())
    db.dump(ids)     
    cli = "shapedb.py -c " + " ".join(map(str,ids))
    print cli

 
