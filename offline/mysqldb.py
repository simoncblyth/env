import MySQLdb
import os


class DBP(dict):
    """
        Reads ini format config file from <path>, storing key,value pairs 
        from <section> into this dict     
        The standard python ConfigParser module is used
            http://docs.python.org/library/configparser.html
        which supports %(name)s style replacements in other values 
 
        Usage example :
             dbp = DBP(path=os.path.expanduser('~/.mydb.cfg') , section="testdb" , envpfx=None )
             print dbp
                 > {'passwd': 'the_db_pass', 'host': '127.0.0.1', 'db': 'the_db', 'user': 'the_db_user'}

        NB as the config file (eg ~/.mydb.cfg ) contains passwords it needs to be treated with care
            * protect it with chmod go-rw (this is enforced )
            * DO NOT COMMIT into any repository 

        If envpfx is supplied and it matches the start of the keys, then the key, value
        pair is exported into the envrionent.

        Example config file with multiple sections to easy config swapping 

[testdb]

; these key names correspond to keyword arguments accepted by MySQLdb.connect
host   : 127.0.0.1
db     : the_db
user   : the_db_user
passwd : the_db_pass

[otherdb]        

host   : 127.0.0.1
db     : otherdb
user   : the_db_user
passwd : the_db_pass

; export these into environment using argument envpfx='ENV_TSQL_' 
; use interpolation to avoid duplication and associated errors
;
ENV_TSQL_URL = 'mysql://%(host)s/%(db)s;'
ENV_TSQL_USER = %(user)s
ENV_TSQL_PSWD = %(passwd)s

[mycascade]

ENV_TSQL_URL = 'mysql://wherever/testdb;mysql://wherever/otherdb'
ENV_TSQL_USER = 'testuser;otheruser'
ENV_TSQL_PSWD = 'testpass;otherpass'


    """
    def __init__(self, path, section, envpfx=None ): 

        ## make sure the config file exists and has appropriate permissions 
        assert os.path.exists( path ), "config path %s does not exist " % path
        from stat import S_IMODE, S_IRUSR, S_IWUSR
        s = os.stat(path)
        assert S_IMODE( s.st_mode ) == S_IRUSR | S_IWUSR , "incorrect permissions, config file must be protected with : chmod go-rw \"%s\" " %  path

        from ConfigParser import ConfigParser
        cfg = ConfigParser()
        cfg.optionxform = str   ## avoid lowercasing keys, making the keys case sensitive
        cfg.read(path)
        secs = cfg.sections()
        assert section in secs , "section %s is not one of these : %s configured in %s " % (section,  secs, path ) 
        for k in cfg.options(section):
            v = cfg.get(section,k)
            self[k] = v
            if envpfx and k.startswith(envpfx):os.environ[k] = v
        self.cfg = cfg
        self.section = section
        self.path = path

 
class DB:
    def __init__(self, **dbp ):
        try:  
            self.conn = MySQLdb.connect( **dbp ) 
            self.dbp = dbp
        except MySQLdb.Error, e: 
            print "Error %d: %s " % ( e.args[0], e.args[1] )
         
    def close(self):
        self.conn.close()

    def execute_(self, cmd):
        cursor = self.conn.cursor(MySQLdb.cursors.DictCursor)
        #cursor = self.conn.cursor()
        cursor.execute( cmd )
        return cursor

    def fetchone(self, cmd ): 
        cursor = self.execute_(cmd)
        row = cursor.fetchone()
        cursor.close()
        return row

    def fetchall(self, cmd ): 
        cursor = self.execute_(cmd)
        rows = cursor.fetchall()
        self.count = cursor.rowcount
        cursor.close()
        return rows

    def __call__(self, cmd):return self.fetchall(cmd)


if __name__=='__main__':
    dbp = DBP( path=os.path.expanduser('~/.mydb.cfg') , section="testdb" , envpfx=None )  
    #print dbp
    db = DB( **dbp )
    rec = db.fetchone("SELECT VERSION()")
    print rec
    for rec in db("SHOW TABLES"):
        print rec 
        tab = rec.values()[0]
        print db("DESCRIBE %s" % tab )


    db.close()

    
