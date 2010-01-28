"""
    ipython enum.py 
         fails as __file__ not defined, instead use
    
    ipython> %run enum.py
    
"""

def path_():
    import os
    return os.path.join( os.path.dirname(__file__), 'test_Conventions.ref' )

class Enum(dict):
    def __init__(self,path=None):
        if not(path):path=path_()
        self.update(eval(file(path).read()))
    def __call__(self, name , v ):
        assert name in self.keys()
        if type(v) == str:
            return self[name].get(v, None)
        elif type(v) == int:
            for k,i in self[name].items():
                if v == i:return k
        return None


if __name__=='__main__':
    e = Enum()
    print e
    for t in e.keys():
        for n,i in e[t].items():
            i = e( t , n ) 
            k = e( t , i )
            assert k == n
            print t,n,i,k

    print e('Site','kDayaBay')
    print e('Site',1)

    assert e('SimFlag','kMC') == 2
    assert e('SimFlag', 2   ) == 'kMC'





   
