
debug_here = lambda : None
try:
    import IPython
    debug_here = IPython.Debugger.Tracer()
except ValueError: 
    pass

## avoid bug in ipython debugger that gives,
##       ValueError: you must specify the default color scheme ... 



def ipy_nosetests(self, arg):
    """
         Hook this up to your ipython environment 
         by adding to ~/.ipython/ipy_user_conf.py :

            try:
                from dybtest.debug import ip_nosetests
                ip.expose_magic('nosetests', ip_nosetests)
            except ImportError:
                pass

          providing the magic command nosetests inside ipython
          that runs with the ipython debugger ready to step in

          Allows :
            In [1]: nosetests tests/test_io.py:test_simo
            ip_nosetests : run -d /data/env/local/dyb/trunk/external/Python/2.5.4/i686-slc4-gcc34-dbg/bin/nosetests -v -s tests/test_io.py:test_simo 


    """
    import os
    path = os.popen("which nosetests").read().rstrip()
    ip = self.api
    cmd = "run -d %s -v -s %s" % ( path , arg  )
    print "ip_nosetests : %s " % cmd
    ip.magic( cmd )


