#!/usr/bin/env python


import mystring
import numpy as np

print('strings for test 1:')
x = np.array(['ABCDE','FGHIJ'],dtype='c')
print("x",x)
print("x.T",x.T)

print("mystring.foo(x)")
mystring.foo(x)

print("mystring.foo(x.T)")
mystring.foo(x.T)

print('strings for test 2:')
_y = ['amsua_n15    ', 'amsua_n18    ', 'amsua_n19    ', 'amsua_metop-a']
y = np.array(_y,dtype='c')

print("y",y)
print("y.T",y.T)

print("mystring.foo(y.T)")
mystring.foo(y.T)



a = np.array(
      [b'TO AB                                                                                           ',
       b'TO BT AB                                                                                        ',
       b'TO BT BR AB                                                                                     ',
       b'TO BT BR BT AB                                                                                  ',
       b'TO BT BR BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT ',
       b'TO SC SC SC SC SC SC SC SC SC BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT DR BT BT BT BT SC ',
       b'TO SC SC SC SC SC SC SC SC SC RE BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT BT BT BR ',
       b'TO SC SC SC SC SC SC SC SC SC RE BT BT BT BT SD                                                 ',
       b'TO SC SC SC SC SC SC SC SC SC SC AB                                                             ',
       b'TO SC SC SC SC SC SC SC SC SC SC BT AB                                                          '], dtype='|S96')

print("a",a)


print("mystring.foo(a)")
mystring.foo(a)

b = a.view("|S1").reshape(-1,96)
#print("mystring.foo(b)")
#mystring.foo(b)

print("mystring.foo(b.T)")
mystring.foo(b.T)







