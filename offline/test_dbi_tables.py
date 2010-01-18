"""
         * use generative nosetests ... to allowing testing of all tables at once
           with out bogging down asserting on one and not getting errors from others
           due to repetitive looping over tables 

 

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

# generative nosetest, yielding separate tests for each table
def test_vld_table_description():
    for table in ["%sVld"%dbi for dbi in dbis]:
        for meth in [m for m in dir(V) if m.startswith('assert_')]:
            yield vld_table_description, table, meth 

# problem here ... it dies on the first assert, nicer to see other failures too for the same table
# so split the test with another param ... 2 param generative ? 
# with 2nd param indicating what is being tested ?

def vld_table_desription(table, meth):
    v = V( db("describe %s" % table ) )
    getattr( v , meth ).__call__(v)  


if __name__=='__main__':

    test_config()
    test_connection()
    test_dbi_pairing()
    test_counts()

    test_vld_table_description()


    db.close()





