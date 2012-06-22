from __future__ import with_statement
import os

def bashrst(path,base,delim="EOU",gbase=None):
   """
   :param path: absolute or relative to base path to bash function file
   :param base: root of the sphinx build, which must contain the `_build` directory 
   :return:  the path to the generated file

   Extract the usage message from the bash function file and writes to a file 
   in `_build/bashrst/`

   Problems:

   #. backticks as needed for rst referencing have special meaning for bash
   
      #. maybe by un-shell-escaping here   
      #. could run the bash function, in order to fill out vars 


   """
   if not gbase:
       gbase = os.path.join(base, "_build", "bashrst" )
   if not path.startswith("/"):
       abspath = os.path.join(base,path)	   
   else:
       abspath = path

   with open(abspath,"r") as fp:
       content = fp.read()

   bits = content.split(delim)
   assert len(bits) == 3, "expect 3 bits delimited by %s in %s not %s " % ( delim, path, len(bits)) 

   relp = abspath[len(base)+1:]
   name,type = os.path.splitext(relp)
   assert type == '.bash', "expecting .bash fileext %s  " % path 
   
   outp = os.path.join( gbase, name + ".rst")
   odir = os.path.dirname(outp)
   if not os.path.exists(odir):
       os.makedirs(odir)

   with open(outp,"w") as fp:
       fp.write(bits[1])

   return outp    



if __name__ == '__main__':
    path = os.path.expandvars("$ENV_HOME/python/python.bash")
    base = os.path.expandvars("$ENV_HOME")
    outp = bashrst(path, base)
    print "wrote %s " % outp


