"""
Hook this up to your ipython environment 
by adding 2 lines to ~/.ipython/ipy_user_conf.py :

  from vdbi import ip_vdbi
  ip.expose_magic('vdbi', ip_vdbi)

providing the magic command vdbi that runs the vdbi app with 
the ipython debugger ready to step in 

Set breakpoints in code by :
 
   from vdbi.dbg import debug_here
   debug_here()    


"""

def ip_vdbi(self, arg):
    import os 
    os.chdir("/tmp")    ## avoid env shadowing 
    path = os.path.join( os.path.dirname( __file__ ), "vdbiapp.py" )
    ip = self.api
    ip.magic("run -d %s" % path )



