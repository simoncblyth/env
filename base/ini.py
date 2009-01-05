
"""

   Replacement for the ailing ini_edit.pl that uses configobj rather
   than reinventing the wheel ...

       python ini.py t.ini "header_logo:alt:smth more sensible"

  Issues :
     1) space removal before a "#"
      
        < ticket_subject_template = $prefix #$ticket.id: $summary
        > ticket_subject_template = $prefix#$ticket.id: $summary
   
      
     2)
       Match failure... hyphen non alpha-numeric
       ERROR skipping triplet account-manager:password_file:/private/etc/apache2/svnsetup/users.conf      
            
                  
"""

import re

class Edit:
    
    """
        NB
            the triplet pattern only matches alphanumeric block/key names in
            order to correctly handle values that incorporate a ":"
            
            "list_values=False, write_empty_values=True" is used to prevent 
            extraeous space additions / blanks becoming ""
            
    """
    tripat = re.compile("(?P<blk>[\w\-]*):(?P<key>[\w\-]*):(?P<val>.*)")
    
    def __init__(self, *args, **kwargs ):
        from configobj import ConfigObj
        self.conf = ConfigObj(*args, **kwargs)
        self.mods = ConfigObj()
        self.merged = False
    
    def write(self):
        if not(self.merged):
            self.merge()
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
            if not(self.mods.has_key(blk)):
                self.mods[blk] = {}
            self.mods[blk][key] = val 
            
    def merge(self):
        self.merged = True
        print "merging %s " % self 
        self.conf.merge(self.mods)
            
    def __repr__(self):
        import pprint
        return "<Edit %s >" % pprint.pformat(self.mods.dict())
    

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
    ed = Edit(args[0], list_values=False, write_empty_values=True )
    import os
    if os.path.isfile(args[0]):
        ed.conf.filename = args[0]
    for t in args[1:]:
        ed(t)
    ed.write()


if __name__=='__main__':
    #test_str()
    import sys
    sys.exit(ini_edit(sys.argv[1:]))
    
    