import re
class Replace:
    """
       Single Pass Multiple Replacement ..
       based on http://code.activestate.com/recipes/81330/
    """
    def __init__(self, dic , **kwa ):
        self.dic = dic 
        self.dic.update(kwa)
        self.pat = "(%s)" % "|".join( map(re.escape, self.dic.keys())  )
    def __call__( self, s ):
        return re.sub( self.pat, lambda m:self.dic[m.group()], s )

if __name__=='__main__':
    r = Replace({'&':'\&' }, hello='hi',world='globe' )
    assert r("hello cruel & world") == 'hi cruel \& globe'
