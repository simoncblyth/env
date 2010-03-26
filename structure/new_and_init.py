"""
   Understanding __new__ and __init__ interplay 
       * http://www.wellho.net/mouth/1146_-new-v-init-python-constructor-alternatives-.html

"""


class A(object):
  """
     Demo returning an instance of class different to itself
  """
  def __new__(cls, *args, **kwds):
    print "one"
    print "A.__new__", args, kwds
    return object.__new__(B, *args, **kwds)
  def __init__(cls, *args, **kwds):
    print "two"
    print "A.__init__", args, kwds




class B(object):
  """
    Demo __new__ calling the __init__ by invokation of __new__ 
    on the superclass 

three
<class '__main__.B'>
<class '__main__.B'>
B.__new__ () {}
four
B.__init__ () {}
<__main__.B object at 0xb7f82bcc>
  """   
  def __new__(cls, *args, **kwds):
    print "three"
    print cls
    print B
    print "B.__new__", args, kwds
    return object.__new__(cls, *args, **kwds)
  def __init__(cls, *args, **kwds):
    print "four"
    print "B.__init__", args, kwds


class C(object):
    """
     The normal way ...

five
C.__init__ () {}
<__main__.C object at 0xb7f57a4c>
    """    
    def __init__(cls, *args, **kwds):
       print "five"
       print "C.__init__", args, kwds



class Other(dict):
    def __init__(self):
        print "Other.__init__ "

class Base(dict):
    """
Base.__new__ 
skip client __init__ by returning Other 
Other.__init__ 
{}
    """
    def __new__(cls, *args, **kwa ):
        print "Base.__new__ "
        if True:
            print "skip client __init__ by returning Other "
            return Other()
        return dict.__new__(cls, *args, **kwa )

    def __init__(self, *args, **kwa ):
        print "Base.__init__ "


class Derived(Base):
    """
        Want the __init__ to do the work of collecting object attributes , 
        but want to skip this work if a persisted version of the object 
        identified by the the kwa is existing 
    """
    def __init__(self, *args, **kwa ):
        print "Derived.__init__ " 




#print C()
#print "====================="
#print A()
#print "====================="
#print B()


print Derived()



