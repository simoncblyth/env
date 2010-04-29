import os
from datetime import datetime
from dbtablecounts import DBTableCounts

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

    # over the persisted instances ... growing with each invocation
    #DBTableCounts._summary()
    # before pulling any persisted instances (which could be very big) 
    # examine the keys of the available instances 
    #s = {}
    #for grp in DBTableCounts._groups():
    #    for d in DBTableCounts._keys(group=grp):
    #        stamp = d.get('stamp', None) ; assert stamp
    #        group = d.get('group', None) ; assert group
    #        if not s.get(stamp,None):s[stamp]=[]
    #        s[stamp].append(group)
    #print s
    #for stamp in sorted(s.keys()):
    #    groups = s[stamp]
    #    if len(groups) == 2:
    #        ia = DBTableCounts( stamp=stamp, group=groups[0] )
    #        ib = DBTableCounts( stamp=stamp, group=groups[1] )            
    #        cf = ia.diff(ib)
    #        if cf:
    #            print "difference found %s %s " % ( stamp,  cf )



