import os
from datetime import datetime
from env.db.dbtablecounts import DBTableCounts

db_name_today = lambda n:"%s_%s" % ( n , datetime.strftime( datetime.now(), "%Y%m%d" ))

def test_recovered_testdb():
    name = 'testdb'
    stamp = datetime.strftime( datetime.now(), "%Y%m%d%H%M%S" )
    db1 = DBTableCounts( group="dybdb1"  , stamp=stamp , db=name )
    rec = DBTableCounts( group="recover" , stamp=stamp , db=db_name_today(name) )
    dif = db1.diff(rec)

    print db1
    print rec 
    assert dif == None, repr(dif)
 

if __name__=='__main__':
    test_recovered_testdb() 


