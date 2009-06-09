"""

    nosetests -v

    Test selection via attributes on the tests :
        nosetests -v -a 'daily'
        nosetests -v -a '!daily'

    Test selection via an expression of the attributes :
        nosetests -v -A "minutes < 11 "
        nosetests -v -A "minutes < 6 "
 

"""

def test_red():pass
def test_green():pass
def test_blue():pass

test_red.minutes = 5 
test_green.minutes = 10
test_blue.minutes = 15

test_blue.daily = True


