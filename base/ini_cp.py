"""
   Reimplementation of the ConfigObj based ~/e/base/ini.py 
   with the less-featureful ConfigParser in order to work with ";" littered 
   Supervisor ini files

       python ini_cp.py t.ini "header_logo:alt:smth more sensible"
      
"""

import re
import sys


def triplet_pattern(delim=":"):
    tripat_s = "(?P<blk>[\w\-]*)\%s(?P<key>[\w\-\.\*]*)\%s(?P<val>.*)"
    return re.compile(tripat_s % ( delim,delim ))

class IniEdit:
    """
        NB
            the triplet pattern only matches alphanumeric (or hyphen) block/key names in
            order to correctly handle values that incorporate a ":"
            
            this collect mods then merge approach has the disadvantage of 
            no easy way to delete a block ... doing things directly would
            be simple and would allow a triplet 
            like "logging::" to cause block deletion
            
            
    """
    def __init__(self, *args, **kwargs ):
        """
            
           A copy of the original settings is made by  
           temporarily setting the filename to None 
           to avoid writing the file 
           
           Setting outfile=None does not to return the lines 
        
        """
        from ConfigParser import ConfigParser
	self.delim = kwargs.pop('delim', ':')
        self.tripat = triplet_pattern(self.delim)

        self.path = path = args[0]

        self.conf = conf = ConfigParser()
        conf.read(path)
  
        self.orig = orig = ConfigParser()
        orig.read(path)  

 
    def write(self, fp=sys.stdout):
         fp = open(self.path,"w")
         return self.conf.write(fp)
    
    def __call__(self, triplet ):
        r = self.tripat.match(triplet)

        conf = self.conf
        if r == None:
            print "env/base/ini.py::IniEdit ERROR skipping triplet %s as failed to match pattern %s " % (triplet, self.delim ) 
        else:
            blk = r.group('blk')
            key = r.group('key')
            val = r.group('val')
            #print "[%s][%s][%s]" % ( blk, key, val )
            if not(conf.has_section(blk)):
                conf.add_section(blk)
            
            ## blank key causes block deletion
            if key == "":
                conf.remove_[blk] = {}
            else:
                if val.find("@DELETE") > -1: 
                    if conf.has_section(blk) and conf.has_option(blk,key): 
                        conf.remove_option(blk,key)
                else:
                    conf.set(blk,key,val) 
        
    def __str__(self):
        from cStringIO import StringIO
        s = StringIO()
        self.conf.write(s)
        return s.getvalue()
                                                      
    def __repr__(self):
        from difflib import unified_diff as ud 
        return "%s" % "\n".join( ud( str(self.orig) , str(self.conf)) )
    

demo = """

#
# a comment

[changeset]
max_diff_bytes = 10000000
max_diff_files = 0
wiki_format_messages = true

[header_logo]
alt = (please configure the [header_logo] section in trac.ini)
height = -1
link =
src = site/your_project_logo.png
width = -1

[inherit]
plugins_dir =
templates_dir =


"""


def ini_edit(args):
    import os    
    delim = os.environ.get('INI_TRIPLET_DELIM',':')
    ie = IniEdit(args[0],  delim=delim )
    for t in args[1:]:
        ie(t)
    ie.write()


if __name__=='__main__':
    import sys
    sys.exit(ini_edit(sys.argv[1:]))
    
    
