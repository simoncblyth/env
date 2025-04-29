#!/usr/bin/env python

from Fortran import chararraytest


test_data = (    '',
                 'text',
                 [('1',), ('2',), ('3',)],
                 ['',''] )

print(chararraytest.chararrayin.__doc__)
for test_str in test_data:
    print('calling chararrayin with args:', repr(test_str))
    chararraytest.chararrayin(test_str)
pass


