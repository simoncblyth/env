"""

  working for simple counting ...
       python runner.py "python count.py 10" 10

  and script running 
       python ~/env/test/runner.py "python share/geniotest.py" 60

   shell commands giving no output though 
       python runner.py "cat runner.py" 10

     possibly 
        1) they complete, before the runner tries to read 
        2) buffering/flushing issue

forked process 88933 for <Run "cat runner.py" rc:None timeout:10 dur:None >  
 completed  <Run "cat runner.py" rc:0 timeout:10 dur:None > 

"""


from match import Matcher, checks
from run import Run

defaults = { 'slow':False , 'timeout':5 , 'verbose':True , 'select_timeout':-1. }

m = Matcher( checks, verbose=defaults['verbose'] )



def test_count3(): Run( "python count.py %s" % 3 , parser=m , opts=defaults  )().assert_()        
def test_count9(): Run( "python count.py %s" % 9 , parser=m , opts=defaults  )().assert_()        


if __name__=='__main__':
    
    from optparse import OptionParser
    import sys
    opr = OptionParser()
    opr.add_option( "--slow"   ,  dest="slow"    , action="store_true" , help="homegrown approach to subprocess running/piping  " )
    opr.add_option( "--verbose" , dest="verbose" , action="store_true" , help="extra output " )
    opr.add_option( "--timeout" , dest="timeout" , type=float  , help="subprocess timeout in seconds, after which it gets killed" )
    opr.add_option( "--select_timeout" , dest="select_timeout" , type=float  , help="select timeout in seconds, interpret as None if negative" )
    opr.set_defaults( **defaults ) 
    (opts , args) = opr.parse_args(sys.argv[1:])    
     
    
    if len(args)<1:
        print sys.modules[__name__].__doc__
        sys.exit(1)
    
    m = Matcher( checks, verbose=opts.verbose )
    r = Run( args[0] , parser=m , opts=opts.__dict__ )        
    r()
    r.assert_()




