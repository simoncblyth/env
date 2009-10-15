"""
Hook this up to your ipython environment 
by adding 2 lines to ~/.ipython/ipy_user_conf.py :

  !vi ~/.ipython/ipy_user_conf.py

  from vdbi.rum.ip_vdbiq import ip_vdbiq
  ip.expose_magic('vdbiq', ip_vdbiq)

providing the magic command vdbiq that runs query.py 
with the ipython debugger ready to step in 

Set breakpoints in code by :
 
   from vdbi.dbg import debug_here
   debug_here()    

"""


def ip_vdbiq(self, arg):
    import os 
    path = os.path.abspath( os.path.join( os.path.dirname( __file__ ), "query.py" ) )
    os.chdir("/tmp")    ## avoid env shadowing 
    ip = self.api
    cmd = "run -d %s" % path
    print "ip_vdbiq : \"%s\" " % cmd
    ip.magic( cmd )
