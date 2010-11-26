"""
    Optimizing creation of numpy arrays from mysql queries 

    Summary findings (details in NOTES.txt )

       * MySQLdb 1.2.3 cursor usage leaky and slow (factor 10) , and cannot be push to big queries (1M rows)
       * 1.2.3 _mysql/fromiter is fast and convenient
       * cythoning can double the _mysql speed at cost of static numpy array types

"""
import sys
from env.npy.scan import ScanIt, ClassTimr        
from tech import Tech
from env.dyb.db import Qry

class LimitScan(ScanIt):
    steps = 5 
    max   = 1000000   

class DebugScan(ScanIt):
    steps = 5 
    max = 10000

def test_scan():
    q = Qry( "DcsPmtHv" , read_default_group="client" , limit="*" , verbose=1, method=1 )
    for n, tc  in enumerate(Tech.classes()):
        #if len(sys.argv) > 1 and int(sys.argv[1]) != n:continue  
        ls = DebugScan(q)
        print "starting query scan %s " % repr(ls)
        for _ in ls:           ## cranking the scan iterator changes parameters
            ct = ClassTimr(tc)
            ct(q)
            ls( **ct )   ## record result in the scan structure at this cursor point 
            pass
        print repr(ls.scan) 
        ls.save()

test_scan.__test__ = True ## tis too slow to run everytime



if __name__ == '__main__':
    test_scan()


