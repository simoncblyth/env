
class A(object):
    def __init__(self):
        print "A",self
        super(A, self).__init__()
    def __repr__(self):
        return "<%s>" % self.__class__
class B(A):
    def __init__(self):
        print "B",self
        super(B, self).__init__()
class C(B):
    def __init__(self):
        print "C",self
        super(C, self).__init__()



if __name__=='__main__':
    c = C()


