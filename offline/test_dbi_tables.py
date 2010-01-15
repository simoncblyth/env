"""
    ideas 
         * try to use generative nosetests ... to allowing testing of all tables at once
           with out bogging down asserting on one and not getting errors from others
           due to repetitive looping over tables 

 

"""

import os
from mysqldb import DB, DBP

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

# example of a generative nose test, generating a separate test for each table
def test_vld_tables():
    for t in ["%sVld"%dbi for dbi in dbis]:
        yield vld_table_description, t 

def vld_table_description(t):
    print t 
    from vld import V
    v = V( db("describe %s" % t ) )
    v.assert_()



if __name__=='__main__':

    test_config()
    test_connection()
    test_dbi_pairing()
    test_counts()

    test_vld_tables()


    db.close()





