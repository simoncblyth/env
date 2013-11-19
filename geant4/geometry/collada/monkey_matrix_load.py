#!/usr/bin/env python
"""
Import this in a prominent location just after the collada import 
as show below.  

This diddles the pycollada transformation matrix loading 
so as to invert the rotation portion of the matrix.

Once this is understood, it should probably be baked into the 
collada file : so visualisation can benefit.

Usage::

    import numpy 
    import collada
    from monkey_matrix_load import _monkey_matrix_load
    collada.scene.MatrixTransform.load = staticmethod(_monkey_matrix_load)

"""

assert 0, "NOV 18 2013 : NO LONGER REQUIRED NOW THAT THE INVROT IS DONE TO THE SOURCE DAE " 
assert 0, "NOV 18 2013 : NO LONGER REQUIRED NOW THAT THE INVROT IS DONE TO THE SOURCE DAE " 
assert 0, "NOV 18 2013 : NO LONGER REQUIRED NOW THAT THE INVROT IS DONE TO THE SOURCE DAE " 

import collada, numpy

def _monkey_matrix_load(_collada,node, diddle=True):
    """
    Avoid changing pycollada in multiple places by monkey patching 
    just the matrix loading to diddle the matrix. 
    The matrix diddling just inverts the rotation portion.

    After doing this it is wring to do the primfix too.

    The advantage over primfix, is that this way also works 
    appropriately with the recursive Node transformations.

    Could avoid having to use this monkeypatch by doing this diddle
    within G4DAEWrite, ie writing diddled to the .dae
    """
    floats = numpy.fromstring(node.text, dtype=numpy.float32, sep=' ')
    if diddle:
        original = floats.copy()
        original.shape = (4,4)
        matrix = numpy.identity(4)
        matrix[:3,:3] = original[:3,:3].T   # transpose/invert the 3x3 rotation portion
        matrix[:3,3] = original[:3,3]       # tack back the translation
        floats = matrix.ravel()
    pass 
    return collada.scene.MatrixTransform(floats, node)



