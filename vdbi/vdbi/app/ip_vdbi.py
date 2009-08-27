"""
Hook this up to your ipython environment 
by adding 2 lines to ~/.ipython/ipy_user_conf.py

from vdbi import ip_vdbi
ip.expose_magic('vdbi', ip_vdbi)

providing the magic command vdbi that runs the vdbi app with 
the ipython debugger ready to step in 

"""

def ip_vdbi(self, arg):
    ip = self.api
    import vdbi, os
    path = os.path.join( os.path.dirname( vdbi.__file__ ), "app" , "vdbiapp.py" )     
    ip.magic("run -d %s" % path )



