#!/usr/bin/env python
"""
   When invoked as a script determines if the 
   configuration named in the single argument exists.

   Usage example : 
      python %(path)s configname  && echo configname exists || echo no configname

"""

import os
class DBConf(dict):
    """
        Reads ini format config file from <path>, storing key,value pairs 
        from <section> into this dict. For example using default <path> 
        of ~/.my.cnf :

          [testdb]
          host      = dybdb1.ihep.ac.cn
          database  = testdb
          user      = dayabay
          password  = youknowoit

    
        The standard python ConfigParser module is used
            http://docs.python.org/library/configparser.html
        which supports %(name)s style replacements in other values 
 
        Usage example :
             dbc = DBConf(sect="client", path="~/.my.cnf" ) 

        NB as the config file contains passwords it needs to be treated with care
            * DO NOT COMMIT into any repository 

    """
    defaults = { 
                 'path':"/etc/my.cnf:$SITEROOT/../.my.cnf:~/.my.cnf", 
                 'sect':"offline_db",
                 'host':"%(host)s", 
                 'user':"%(user)s", 
                   'db':"%(database)s", 
                 'pswd':"%(password)s",
                  'url':"mysql://%(host)s/%(database)s", 
                'urlsa':"mysql://%(user)s:%(password)s@%(host)s/%(database)s", 
               'engine':"django.db.backends.mysql",
                 'port':"",
                  'fix':None,
               }

    def Export(cls, sect=None , **extras ):
        """
             for export into env of python process 
             CAUTION : this class method is invoked by the C++ DbiCascader ctor 
        """
        cnf = DBConf( sect=sect )  
        if cnf.fix == None:
            cnf.export_to_env(**extras)
        else:
            from dbcas import DBCas
            cas = DBCas(cnf)
            tas = cas.spawn()
            cnf.export_to_env( supplier=tas )                
        return cnf
        #print dbc.dump_env()
    Export = classmethod( Export )

    def __init__(self, sect=None , path=None , user=None, pswd=None, url=None , host=None, db=None , port=None, engine=None, urlsa=None, fix=None, verbose=False, secure=False, from_env=False ): 
        """

           Documented in the Database/Running section of the Offline User Manual 

           Interpolates the DB connection parameter patterns gleaned 
           from arguments, envvars or defaults (in that precedence order)
           into usable values using the context supplied by the 
           <sect> section of the ini format config file at <path>


            Arguments 
                  sect : section in config file
                  path : path to config file 
                  user : username 
                  pswd : password
                  url  : connection url
                  host : db host 
                  db   : db name
                  fix  : triggers fixture loading into tmporary 
                         spawned cascade and specifies paths to fixture files
                         for each member of the cascade (semi-colon delimited)  

            Envvars
                 DBCONF        points to section in config file 
                 DBCONF_PATH   accept a colon delimited list of paths 
                 DBCONF_USER
                 DBCONF_PWSD
                 DBCONF_URL
                 DBCONF_HOST
                 DBCONF_DB
                 DBCONF_FIX

            The "DBCONF" envvar existance also triggers the 
            DybPython.DBConf Export in the DbiCascader.cxx code

            The DBCONF_PATH is a colon delimited list of paths that are 
            user (~) and $envvar OR ${envvar} expanded, some of the paths 
            may not exist.  When there are repeated settings in more than one
            file the last one wins.

            In secure mode a single protected config file is required, the security 
            comes with a high price in convenience

        """

        self.secure = secure
        self.verbose = verbose

        sect   = sect    or os.environ.get('DBCONF',      DBConf.defaults['sect'] ) 
        path   = path    or os.environ.get('DBCONF_PATH', DBConf.defaults['path'] ) 
        user   = user    or os.environ.get('DBCONF_USER', DBConf.defaults['user'] ) 
        pswd   = pswd    or os.environ.get('DBCONF_PSWD', DBConf.defaults['pswd'] ) 
        url    = url     or os.environ.get('DBCONF_URL',  DBConf.defaults['url'] ) 
        host   = host    or os.environ.get('DBCONF_HOST', DBConf.defaults['host'] ) 
        db     = db      or os.environ.get('DBCONF_DB'  , DBConf.defaults['db'] ) 
        fix    = fix     or os.environ.get('DBCONF_FIX' , DBConf.defaults['fix'] ) 
        ## for django
        port   = port    or os.environ.get('DBCONF_PORT' ,  DBConf.defaults['port'] ) 
        engine = engine  or os.environ.get('DBCONF_ENGINE' , DBConf.defaults['engine'] ) 
        ## for SQLAlchemy
        urlsa   = urlsa  or os.environ.get('DBCONF_URLSA',  DBConf.defaults['urlsa'] ) 
  
        if self.secure:
            self._check_path( path )
        if not from_env:
            self.configure( sect, path ) 
 
        self.sect = sect
        self.path = path
        self.user = user
        self.pswd = pswd
        self.url  = url
        self.host = host
        self.db   = db
        self.fix  = fix
        self.port = port
        self.engine = engine
        self.urlsa = urlsa

    def mysqldb_parameters(self):
        #return dict(read_default_file=self.path, read_default_group=self.sect)
        d = dict(host=self.host % self, user=self.user % self, passwd=self.pswd % self, db=self.db % self ) 
        if self.verbose:
            print "dbconf : connecting to %s " % dict(d, passwd="***" )
        return d
    mysqldb = property( mysqldb_parameters ) 
 
    def django_parameters(self):
        d = dict( ENGINE=self.engine % self, NAME=self.db % self , USER=self.user % self, PASSWORD=self.pswd % self, HOST=self.host % self , PORT=self.port % self )
        if self.verbose:
            print "dbconf : connecting to %s " % dict(d, PASSWORD="***" )
        return d   
    django = property( django_parameters ) 
 
    def sqlalchemy_url(self):
        if self.verbose:
            print "dbconf : connecting to %s " % dict(self, password="***" )
        return self.urlsa % self   
    sqlalchemy = property( sqlalchemy_url ) 
        

    def _check_path(self, path ):
        """
              Check existance and permissions of path 
        """ 
        assert os.path.exists( path ), "config path %s does not exist " % path
        from stat import S_IMODE, S_IRUSR, S_IWUSR
        s = os.stat(path)
        assert S_IMODE( s.st_mode ) == S_IRUSR | S_IWUSR , "incorrect permissions, config file must be protected with : chmod go-rw \"%s\" " %  path

    def configure( self, sect, path ):
        cfp, paths = DBConf.read_cfg( path )
        secs = cfp.sections()
        print "dbconf : reading config from section \"%s\" obtained from %s (last one wins)  " % ( sect, repr(paths) ) 
        assert sect in secs  , "section %s is not one of these : %s configured in %s " % ( sect,  secs, paths ) 
        self.update( cfp.items(sect) )

    def dump_env(self, epfx='env_'):
        e = {}
        for k,v in os.environ.items():
            if k.startswith(epfx.upper()):e.update({k:v} )   
        return e


    urls  = property( lambda self:(self.url  % self).split(";") )
    users = property( lambda self:(self.user % self).split(";") )
    pswds = property( lambda self:(self.pswd % self).split(";") )
    fixs  = property( lambda self:(self.fix  % self).split(";") )

    def from_env(cls):
        """
            could also reconstruct from a live gDbi.cascader
        """
        url  = os.environ.get( 'ENV_TSQL_URL', None )
        user = os.environ.get( 'ENV_TSQL_USER', None )
        pswd = os.environ.get( 'ENV_TSQL_PSWD', None )
        assert url and user and pswd , "DBConf.from_env reconstruction requites the ENV_TSQL_* "
        cnf = DBConf(url=url, user=user, pswd=pswd,from_env=True)
        return cnf
    from_env = classmethod(from_env)


    def export_(self, **extras):
        """
            Exports the interpolated DBCONF_* into 
            corresponding envvars :

                ENV_TSQL_*   for access to DBI tables via DatabaseInterface 
                DYB_DB_*     for access to non-DBI tables via DatabaseSvc

        """ 
        supplier = extras.pop('supplier', None )
        if supplier:
            print "export_ supplier is %s " % supplier
        else:
            supplier = self

        self.export={}
	self.export['ENV_TSQL_URL'] =  supplier.url  % self 
        self.export['ENV_TSQL_USER'] = supplier.user % self 
        self.export['ENV_TSQL_PSWD'] = supplier.pswd % self 

        self.export['DYB_DB_HOST']   = supplier.host % self
        self.export['DYB_DB_NAME']   = supplier.db   % self
        self.export['DYB_DB_USER']   = supplier.user % self
        self.export['DYB_DB_PSWD']   = supplier.pswd % self
      
        for k,v in extras.items():
            self.export[k] = v % self 


    def read_cfg( cls , path=None ):
        """
              Read section from the config into self
        """
        path = path or os.environ.get('DBCONF_PATH', DBConf.defaults['path'] ) 
        from ConfigParser import ConfigParser
        cfp = ConfigParser(DBConf.prime_parser())
        cfp.optionxform = str   ## avoid lowercasing keys, making the keys case sensitive
        paths = cfp.read( [os.path.expandvars(os.path.expanduser(p)) for p in path.split(":")] )   
        return cfp, paths
    read_cfg = classmethod( read_cfg )


    def has_config( cls , name ):
        """ return if the named config is available in any of the available DBCONF files """ 
        cfp, paths = DBConf.read_cfg()
        return name in cfp.sections() 
    has_config = classmethod( has_config ) 

    def prime_parser( cls ):
        """
             prime parser with "today" to allow expansion of 
              %(today)s in ~/.my.cnf
             allowing connection to a daily recovered database named after todays date
        """
        from datetime import datetime
        return dict(today=datetime.now().strftime("%Y%m%d"))
    prime_parser = classmethod( prime_parser )


    def export_to_env(self, **extras):
        self.export_(**extras)
        print "dbconf:export_to_env from %s section %s " % ( self.path, self.sect ) 
        os.environ.update(self.export) 
        if self.verbose:
            print " ==> %s " % dict(self.export, ENV_TSQL_PSWD='***', DYB_DB_PSWD='***' ) 


if __name__=='__main__':
    import sys
    assert len(sys.argv) == 2 , __doc__ % { 'path':sys.argv[0] }
    sys.exit( not(DBConf.has_config(sys.argv[1])) )

