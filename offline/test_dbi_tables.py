"""

   In addition to standard nosetests the below makes use of 
   a generative nosetest 'test_vld_description' which 
   allowing testing of all (tables,'assert_'s) in a single nosetest run
           
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

db = dbp = dbis = None

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
            r = db.fetchone("select count(*) from %s" % t )    
            print "%-20s %s " % ( t, r.values()[0] )

# generative nosetest, yielding matrix of tests for each (table, assert_ method)
def test_vld_description():
    for table in ["%sVld"%dbi for dbi in dbis]:
        for meth in [m for m in dir(V) if m.startswith('assert_')]:
            yield vld_description, table, meth 

def vld_description(table, meth):
    v = V( db("describe %s" % table ) )
    getattr( v , meth )()  


if __name__=='__main__':

    test_config()
    test_connection()
    test_dbi_pairing()
    test_counts()

    test_vld_description()


    db.close()





