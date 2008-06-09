#!/usr/bin/env python

"""

  Based on 
      http://svn.python.org/projects/distutils/trunk/misc/get_metadata.py

   Pry open a setup script and pick out the juicy bits, ie. the
   distribution meta-data.

   And try to find the name of the egg

"""

import sys
from distutils.core import run_setup

class Setup:
    def __init__(self, path ):
        self.path = path
        self.dist = run_setup( path, script_args=[], stop_after="init")
            
    def dump(self):
        dist = self.dist
        print """\
%s is the setup script for %s; description:
%s

contact:  %s <%s>
info url: %s
licence:  
""" % (self.path, dist.get_fullname(), dist.get_description(),
       dist.get_contact(), dist.get_contact_email(),
       dist.get_url() )
    

    def egg_name(self):
        """
            http://svn.python.org/projects/sandbox/branches/setuptools-0.6/setuptools/command/bdist_egg.py  
        
           unfortunately does not do the right thing with native eggs
        """
        cmd = self.dist.get_command_obj("bdist_egg")
        cmd.dist_dir=""
        cmd.finalize_options()
        return cmd.egg_output 

def main(args):
    s = Setup(args[1])
    print s.egg_name()



if __name__ == "__main__":
    sys.exit(main(sys.argv))
