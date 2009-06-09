"""

nosetests -v
test_simple.test_red ... ok
test_simple.test_green ... ok
test_simple.test_blue ... ok
----------------------------------------------------------------------
Ran 3 tests in 0.334s
OK


nosetests -v -a 'daily'
test_simple.test_blue ... ok
----------------------------------------------------------------------
Ran 1 test in 0.038s
OK


nosetests -v -a '!daily'
test_simple.test_red ... ok
test_simple.test_green ... ok
----------------------------------------------------------------------
Ran 2 tests in 0.036s
OK
  

"""

def test_red():pass
def test_green():pass
def test_blue():pass

test_blue.daily = True


