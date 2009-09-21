"""
Hook this up to your ipython environment 
by adding 2 lines to ~/.ipython/ipy_user_conf.py :

  from plvdbi import ip_plvdbi
  ip.expose_magic('plvdbi', ip_plvdbi)

providing the magic command plvdbi that runs the plvdbi app with 
the ipython debugger ready to step in 

Set breakpoints in code by :
 
   from vdbi import debug_here
   debug_here()    


"""

def ip_plvdbi(self, arg):
    import os 
    
    paster = os.popen("which paster").read().rstrip()
    ini = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__) ) ), "development.ini" )
    
    ip = self.api
    ip.magic("run -d %s serve %s " % ( paster , ini ) )
