"""

  working for simple counting ...
       python runner.py "python count.py 10" 10

  and script running 
[blyth@cms01 RootIOTest]$ python ~/env/test/runner.py "python share/geniotest.py" 60



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
import sys
   
m = Matcher( checks, verbose=True )

def test_count3(): Run( "python count.py %s" % 3 , parser=m , timeout=5  )().assert_()        
def test_count9(): Run( "python count.py %s" % 9 , parser=m , timeout=5  )().assert_()        

if __name__=='__main__':
    m = Matcher( checks, verbose=True )
    r = Run( sys.argv[1] , parser=m , timeout=int(sys.argv[2]))        
    r()
    r.assert_()




