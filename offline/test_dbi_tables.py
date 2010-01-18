"""

   In addition to standard nosetests the below makes use of 
   a generative nosetests such as 'test_vld_description' which 
   allow testing of all (tables,'assert_'s) in a single nosetest run
           
    Notes for environment for running tests on WW
 
        * need source python and mysql env setup in .bash_profile
        * thence use my virtual python (which includes nose)  
             * {{{vip-;vip-dbi}}}
        * allowing to run tests
             * {{{nosetests -v test_dbi_tables.py}}}

"""

import os
from mysqldb import DB, DBP
from vld import V
from enum import Enum
e = Enum()

db = dbp = dbis = None

def vlds():return ["%sVld"%dbi for dbi in dbis]

def test_config():
    global dbp
    dbp = DBP( path=os.path.expanduser('~/.dybdb.ini') , section="testdb" , envpfx=None )  
    #print dbp
    
def test_connection():
    global db
    db = DB( **dbp )
    print db.fetchone("SELECT VERSION()")

def test_dbi_pairing():
    ## check all *Vld tables have corresponding payload * table 
    vlds = [r.values()[0][:-3] for r in db("SHOW TABLES LIKE '%Vld'")]
    pays = [r.values()[0] for r in db("SHOW TABLES") if r.values()[0] in vlds]
    assert len(vlds) == len(pays) , "DBI validity %s and payload %s table mismatch " % ( vlds , pays )
    global dbis  
    dbis = vlds 

def test_counts():
    for dbi in dbis:
        for t in (dbi, "%sVld" % dbi ):
            c = db.fetchcount("select count(*) from %s" % t )    
            print "%-20s %s " % ( t, c )

def test_vld_description():
    for table in ["%sVld"%dbi for dbi in dbis]:
        for meth in [m for m in dir(V) if m.startswith('assert_')]:
            yield vld_description, table, meth 
def vld_description(table, meth):
    v = V( db("describe %s" % table ) )
    getattr( v , meth )()  


def test_vld_enumfield():
    for table in vlds():
        for field in V.enumfields.keys():yield vld_enumfield, table, field 
def vld_enumfield( table, field ):
    v = tuple(e[V.enumfields[field]].values())
    q = "select count(*) from %s where %s not in %s " % ( table , field , v )  
    assert db.fetchcount(q) == 0 , " field with non-enumerated values, query: \"%s\" " % q 


def test_vld_datetimefield():
    for table in vlds():
        for field in V.datefields:yield vld_datetimefield, table, field 
def vld_datetimefield( table , field ):
    q = "select count(*) from %s where %s <> NULL and %s < FROM_UNIXTIME(0) " % ( table , field, field ) 
    assert db.fetchcount(q) == 0 , " invalid non-null datetime, query: \"%s\" " % q 



if __name__=='__main__':

    test_config()
    test_connection()
    test_dbi_pairing()
    test_counts()

    ## generative tests rely on nose knowing how to run them ... so these calls are useless
    #test_vld_description()
    #test_vld_enumfield()
    #test_vld_datetimefield()

    db.close()





