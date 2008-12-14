"""
  Hands-off testing of python scripts ... the scripts are run in a subprocess
  while a parser examines the stdout piped out of the subprocess, against regexps
  managed by the matcher.

"""

from dybtest import Matcher, Run

checks = {
          '.*ERROR':1,
          '.*FATAL':2,
          '.*\*\*\* Break \*\*\* segmentation violation':3, 
          '.*IOError':4,
          '^\#\d':None 
        }

Run.parser = Matcher( checks, verbose=False )
Run.opts = { 'maxtime':300 }

def test_build():
    Run("dyb__.sh mmake")()

def test_simhists():  
    Run( "nuwa.py -n 10 share/simhists.py")()

def test_genhists():  
    Run( "nuwa.py -n 10 share/genhists.py")()


def mean(fname,hname):
    from ROOT import TFile 
    f = TFile(fname)
    return f.Get(hname).GetMean() 

def test_mean_func():
    """
          test_mean_func : 6 lines 
    """
    assert 3 < mean('simhists.root','kineEnergy') < 5

def test_mean_direct():
    """
          test_mean_direct : 5 lines

          try adjusting the range to make it fail and using the -d option  
               nosetests -d -v tests/test_SimHistsExample.py
    """
    from ROOT import TFile
    f = TFile('simhists.root')
    m = f.Get('kineEnergy').GetMean() 
    assert 3 < m < 5




test_build.__test__=False
test_simhists.__test__=False
test_genhists.__test__=False



if __name__=='__main__':
    #test_simhists()
    #test_genhists()

    print mean('simhists.root','kineEnergy')



