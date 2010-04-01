#!/usr/bin/env python
"""

   DBConf is deprecated after discovering similar capabilities
   in MySQLdb  :
      * read_default_file 
      * read_default_group   
       
   Using these allow a single config file  ~/.my.cnf
   to be used from both command line mysql client 
   and python      





   Preview environment setup read from .ini 
       ./dbconf.py --path demo.ini --sect testdb  

   Equip your environment with ENV_TSQL_ envvars 
   by placing the below into your .bash_profile where the section name
   matches a section name in your $HOME/.dybdb.ini
 
       eval $(./dbconf.py --sect testdb  )

"""
import os

class DBConf(dict):
    """
        Reads ini format config file from <path>, storing key,value pairs 
        from <section> into this dict     
        The standard python ConfigParser module is used
            http://docs.python.org/library/configparser.html
        which supports %(name)s style replacements in other values 
 
        Usage example :
             cfg = DBConf(path=os.path.expanduser('~/.dybdb.ini') , sect="testdb" , epfx=None )

        NB as the config file contains passwords it needs to be treated with care
            * protect it with chmod go-rw (this is enforced )
            * DO NOT COMMIT into any repository 

        If epfx is supplied and it matches the start of the keys, then the key, value
        pair is exported into the envriroment.

    """

    def __init__(self, *args, **kwa ):

        path = kwa.get('path', None)
        sect = kwa.get('sect', None)
        epfx = kwa.get('epfx', None)
        src  = kwa.get('src' , None)
        
        from ConfigParser import ConfigParser
        cfg = ConfigParser()
        cfg.optionxform = str   ## avoid lowercasing keys, making the keys case sensitive

        if not(src):
            ## make sure the config file exists and has appropriate permissions 
            assert os.path.exists( path ), "config path %s does not exist " % path
            from stat import S_IMODE, S_IRUSR, S_IWUSR
            s = os.stat(path)
            assert S_IMODE( s.st_mode ) == S_IRUSR | S_IWUSR , "incorrect permissions, config file must be protected with : chmod go-rw \"%s\" " %  path
            cfg.read(path)
        else:
            from StringIO import StringIO
            cfg.readfp( StringIO(src) )

        secs = cfg.sections()
        assert sect in secs , "section %s is not one of these : %s configured in %s " % (sect,  secs, path ) 
        for k in cfg.options(sect):
            v = cfg.get(sect,k)
            self[k] = v
            if epfx and k.startswith(epfx):os.environ[k] = v
    
        self.cfg = cfg
        self.sect = sect
        self.path = path
        self.epfx = epfx

    def __repr__(self):
        return "# ./dbconf.py --path %s --sect %s --epfx %s " % ( self.path, self.sect, self.epfx ) 

    def export(self, shell='bash'):
        epfx = self.epfx
        o = []
        for k,v in self.items():
            if epfx and k.startswith(epfx):o.append( "export %s='%s' " % (k, v) )
        return ";".join(o)



if __name__=='__main__':

    from optparse import OptionParser
    parser = OptionParser()
    defaults = { 'path':os.path.expanduser('~/.dybdb.ini') , 'sect':"testdb" , 'epfx':"ENV_TSQL_" }

    parser.add_option("-p", "--path", dest="path", help="path to ini file, default: %s " % defaults['path'], metavar="PATH")
    parser.add_option("-s", "--sect", dest="sect", help="active section in the ini file, default: %s " % defaults['sect'], metavar="SECT")
    parser.add_option("-e", "--epfx", dest="epfx", help="export envvars with names starting with this prefix, default: %s  " % defaults['epfx'] , metavar="EPFX")
    parser.set_defaults( **defaults )
    (opts, args) = parser.parse_args()

    cfg = DBConf( path=opts.path , sect=opts.sect , epfx=opts.epfx )  
    print cfg.export(), cfg



  
