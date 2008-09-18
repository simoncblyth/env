
def test_evens():
    for i in range(0,5,2):
        yield check_even, i, i*3

def test_odds():
    for i in range(1,6,2):
        yield check_even, i, i*3

#test_evens.__test__=fails

def check_even(n, nn):
    assert n % 2 == 0 or nn % 2 == 0      




