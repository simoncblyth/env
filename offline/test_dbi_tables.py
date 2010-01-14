import os
from mysqldb import DB, DBP

db = dbp = dbis = None

def test_config():
    global dbp
    dbp = DBP( path=os.path.expanduser('~/.mydb.cfg') , section="testdb" , envpfx=None )  
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
       

if __name__=='__main__':
    test_config()
    test_connection()
    test_dbi_pairing()
    test_counts()

    db.close()


