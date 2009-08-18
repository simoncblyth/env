

def tmp_dbname(dbname):
    return "%s_testing" % dbname 

def test_sa_createdb():    
    from django.conf import settings
    dburi = settings.DJANGO_SQLALCHEMY_DBURI    
    
    from sqlalchemy.engine import url
    url = url.make_url( dburi )
    tmpdb = tmp_dbname( url.database )
    url.database = None
    
    from sqlalchemy import create_engine
    engine = create_engine( url )
    connection = engine.connect()
    connection.execute("create database if not exists %s" % tmpdb )
    print "created %s " % tmpdb
    connection.close()

def test_sa_dropdb():
    from django.conf import settings
    dburi = settings.DJANGO_SQLALCHEMY_DBURI
    
    from sqlalchemy.engine import url
    url = url.make_url( dburi )
    tmpdb = tmp_dbname( url.database )
    url.database = None
    
    from sqlalchemy import create_engine
    engine = create_engine( url )
    connection = engine.connect()
    connection.execute("drop database if exists %s" % tmpdb )
    print "dropped %s " % tmpdb
    connection.close()	



def _test_create():
    from django.db import connection
    connection.creation.create_test_db(verbosity=2)

if __name__=='__main__':
    test_sa_createdb()
    test_sa_dropdb()