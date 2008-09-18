
from inspect import getsourcelines 

def a():
     print 'a!'

print getsourcelines(getsourcelines)
print getsourcelines(a)


