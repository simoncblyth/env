
from env.sa import *
from datetime import datetime


session = Session()        
for c in "red green blue cyan magenta yellow".split():
    q, created = get_or_create( session, Qry , name=c ) 
    print q, created
session.flush()


for o in session.query(Qry).all():
    print o


print "count : %d ", session.query(Qry).count()
print session.query(Qry).count()

assert SimPmtSpec.count() == 2546 
assert CalibPmtSpec.count() == 4160 
assert SimPmtSpec.get((1,100)).VERSIONDATE == datetime(2010, 1, 20, 0, 0)   ## CPK get 
assert CalibFeeSpecVld.count() == 111

for v in CalibPmtSpecVld.all():
    print v 


