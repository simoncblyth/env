
"""

   Replacement for the ailing ini_edit.pl that uses configobj rather
   than reinventing the wheel ...

       python ini.py t.ini "header_logo:alt:smth more sensible"

  Issues :
     1) space removal before a "#"
      
        < ticket_subject_template = $prefix #$ticket.id: $summary
        > ticket_subject_template = $prefix#$ticket.id: $summary
   
      
"""

import re


def pp(a):
    import pprint
    return pprint.pformat(a)

class IniEdit:
    
    """
        NB
            the triplet pattern only matches alphanumeric (or hyphen) block/key names in
            order to correctly handle values that incorporate a ":"
            
            "list_values=False, write_empty_values=True" is used to prevent 
            extraeous space additions / blanks becoming ""
            
            
            this collect mods then merge approach has the disadvantage of 
            no easy way to delete a block ... doing things directly would
            be simple and would allow a triplet 
            like "logging::" to cause block deletion
            
            
    """
    tripat = re.compile("(?P<blk>[\w\-]*):(?P<key>[\w\-]*):(?P<val>.*)")
    
    
    @classmethod
    def text(cls, conf):
        fn = conf.filename
        conf.filename = None
        txt = conf.write()
        conf.filename = fn 
        return txt
    
    def __init__(self, *args, **kwargs ):
        """
            
           A copy of the original settings is made by  
           temporarily setting the filename to None 
           to avoid writing the file 
           
           Setting outfile=None does not to return the lines 
        
        """
        from configobj import ConfigObj
        self.conf = ConfigObj( *args, **kwargs)
        self.orig = ConfigObj( self.text( self.conf ))
        
    
    def write(self):
         return self.conf.write()
    
    def __call__(self, triplet ):
        r = self.tripat.match(triplet)
        if r == None:
            print "ERROR skipping triplet %s as failed to match pattern " % triplet 
        else:
            blk = r.group('blk')
            key = r.group('key')
            val = r.group('val')
            print "[%s][%s][%s]" % ( blk, key, val )
            if not(self.conf.has_key(blk)):
                self.conf[blk] = {}
            
            ## blank key causes block deletion
            if key == "":
                self.conf[blk] = {}
            else:
                self.conf[blk][key] = val 
        
    def __str__(self):
        return "\n".join(self.text( self.conf ))
                                                      
    def __repr__(self):
        from difflib import unified_diff as ud 
        return "%s" % "\n".join( ud( self.text(self.orig) , self.text(self.conf)) )
    

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


def test_str():
    from StringIO import StringIO as SIO
    cnf = Edit( SIO(demo) )
    cnf("inherit:plugins_dir:hello")
    print cnf.write()


def ini_edit(args):
    ie = IniEdit(args[0], list_values=False, write_empty_values=True )
    import os
    if os.path.isfile(args[0]):
        ie.conf.filename = args[0]
    for t in args[1:]:
        ie(t)
    ie.write()


if __name__=='__main__':
    #test_str()
    import sys
    sys.exit(ini_edit(sys.argv[1:]))
    
    