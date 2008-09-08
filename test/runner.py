"""

  Usage examples :
  
        python runner.py "python count.py 10" 
        python runner.py "cat runner.py" 
        python ~/env/test/runner.py "python share/geniotest.py" 
    
        python runner.py --help
    
"""

from match import Matcher, checks
from run import Run

opts={'maxtime':5 , 'verbose':True } 

m = Matcher( checks, verbose=opts['verbose'] )

def test_count3(): Run( "python count.py %s" % 3 , parser=m , opts=opts  )().assert_()        
def test_count9(): Run( "python count.py %s" % 9 , parser=m , opts=opts  )().assert_()        


if __name__=='__main__':
    
    from optparse import OptionParser
    import sys
    opr = OptionParser()
    opr.add_option( "--verbose" , dest="verbose" , action="store_true" , help="extra output " )
    opr.add_option( "--maxtime" , dest="maxtime" , type=float          , help="subprocess time to live in seconds, after which it gets killed" )
    opr.add_option( "--timeout" , dest="timeout" , type=float          , help="select timeout in seconds, interpret as None if negative" )
    opr.set_defaults( **opts ) 
    (conf , args) = opr.parse_args(sys.argv[1:])    
     
    if len(args)<1:
        print sys.modules[__name__].__doc__
        sys.exit(1)
    
    r = Run( args[0] , parser=m , opts=conf.__dict__ )().assert_()        



