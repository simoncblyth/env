"""

  Testing with 
        nosetests test_exits.py -vv 

   sys.exit are "caught" cleanly, reported as ERROR, with subsequent tests proceeding
   failing asserts, reported as FAIL
   os._exit just exits, with no status reported, preventing running of subsequent tests
   and preventing the test result reporting for all the tests  
        (just like segmentation faults)  

   Try out the InsulateRunner 
       nosetests test_exits.py -vv --with-insulate

   this nose plugin avoids the exit problem (it uses master/slave nosetest processes) 
   sending tests over to the slave one by one, and collecting the output for reporting
   by the master
   
   It returns ERROR for the os._exit with a CrashInTestError exception being reported,
   UNFORTUNATELY: does not provide the crashing output .. but nevertheless a great 
   improvement that all tests are run and reported on 

   Does it work in concert with xml reporting ?
   Using this method works fine :   
      nosetests test_exits.py -vv --with-insulate --with-xml-output 2> out.xml
      
   But with this way :
      nosetests test_exits.py -vv --with-insulate --with-xml-output --xml-outfile=out.xml
   using "--xml-outfile=out.xml" only the last test is reported in the out.xml, maybe the
   slave is overwriting the output as it gets instructed to do each test by the master, 
   so have to prevent the slave from seeing the "--xml-outfile=out.xml" option by :     
      
  --insulate-skip-after-crash
  --insulate-not-in-slave=NOT_IN_SLAVE
  --insulate-in-slave=IN_SLAVE
  --insulate-show-slave-output

      nosetests test_exits.py -vv --with-insulate --with-xml-output --insulate-not-in-slave="--xml-outfile=out.xml"
              this doesnt work does not provide out.xml
              
  Instead must duplicate (precisely) the option to be given to the insulate-master 
  and not given to the insulate-slave :

      nosetests test_exits.py -vv --with-insulate --with-xml-output --xml-outfile=out.xml --insulate-not-in-slave="--xml-outfile=out.xml"
               works as desired 
     
             
    Another issue with Gaudi test running ... is fixture interference : where a test that does not 
    cause python to die but that leaves Gaudi is a screwed state causing subsequent tests to fail,
    whereas they would have succeeded if they were done separately.  The safest approach is to do 
    every test in a fresh process, avoiding interference. 
    
    
    A small modfification to insulate provides additional option to run every test in a new slave : 
         nosetests test_exits.py -vv --with-insulate --insulate-every-test
    
    
"""


import sys
import os

def test_pass_1():pass
def test_exception():raise Exception
def test_pass_2():pass
def test_assert():assert False
def test_pass_3():pass
def test_sysexit0():sys.exit(0)
def test_pass_4():pass
def test_sysexit1():sys.exit(1)
def test_pass_5():pass
def test_osexit0():os._exit(0)
def test_pass_6():pass
def test_osexit1():os._exit(1)
def test_pass_7():pass

