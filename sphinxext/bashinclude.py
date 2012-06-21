from __future__ import with_statement
import os

from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.misc import Include as BaseInclude


def bashrst(path,base,delim="EOU"):
   """
   :param path: absolute path to bash function file
   :param base: root of the sphinx build, which must contain the `_build` directory 
   :return:  the path to the generated file

   Extract the usage message from the bash function file and writes to a file 
   in `_build/bashrst/`

   """
   with open(path,"r") as fp:
       content = fp.read()

   bits = content.split(delim)
   assert len(bits) == 3, "expect 3 bits delimited by EOU in %s " % path 

   relp = path[len(base)+1:]
   name,type = os.path.splitext(relp)
   assert type == '.bash', "expecting .bash fileext %s  " % path 
   
   outp = os.path.join(base, "_build", "bashrst", name + ".rst")
   odir = os.path.dirname(outp)
   if not os.path.exists(odir):
       os.makedirs(odir)

   with open(outp,"w") as fp:
       fp.write(bits[1])
   return outp    


class BashInclude(BaseInclude):
    """
    Like the standard "Include" directive, but extracts the 
    usage message string from a collection of bash functions  
    """
    def run(self):
       	env = self.state.document.settings.env
	rel_filename, filename = env.relfn2path(self.arguments[0])
	self.arguments[0] = bashrst(filename, os.getcwd() )
	return BaseInclude.run(self)


def setup(app):
    app.add_directive('bashinclude', BashInclude)



if __name__ == '__main__':
    path = os.path.expandvars("$ENV_HOME/python/python.bash")
    base = os.path.expandvars("$ENV_HOME")
    outp = bashrst(path, base)
    print "wrote %s " % outp


