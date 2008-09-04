"""

  working for simple counting ...
       python runner.py "python count.py 10" 10

   shell commands giving no output though
       python runner.py "cat runner.py" 10

forked process 88933 for <Run "cat runner.py" rc:None timeout:10 dur:None >  
 completed  <Run "cat runner.py" rc:0 timeout:10 dur:None > 



"""

from match import Matcher
from run import Run
import sys

if __name__=='__main__':
    checks = {
        '.*FATAL':1,
        '.*ERROR':2,
     }
    m = Matcher( checks, verbose=True )
    r = Run( sys.argv[1] , parser=m )        
    r(timeout=int(sys.argv[2]))




