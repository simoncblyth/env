#!/usr/bin/python
"""
* http://stackoverflow.com/questions/2182002/convert-big-endian-to-little-endian-in-c-without-using-provided-func

::

    0xff000000 --> 0x000000ff 
    0x00ff0000 --> 0x0000ff00 
    0x0000ff00 --> 0x00ff0000 
    0x000000ff --> 0xff000000 
    0xdeadbeef --> 0xefbeadde 

// move byte 3 to byte 0
// move byte 1 to byte 2
// move byte 2 to byte 1
// byte 0 to byte 3

"""


class B(object):
    """
    uvec4 b = uvec4( 0xff, 0xff00, 0xff0000,0xff000000) ;
    """
    x = 0xff
    y = 0xff00
    z = 0xff0000
    w = 0xff000000

class R(object):
    """
    uvec4 r = uvec4( 
    """
    def __init__(self, n):
        self.n = n
    x = property(lambda self:self.n>>24)
    y = property(lambda self:self.n>>8)
    z = property(lambda self:self.n<<8)
    w = property(lambda self:self.n<<24)
    

uint32_swap0 = lambda x:((x>>24) & 0xff) | ((x<<8) & 0xff0000) | ((x>>8) & 0xff00) | ((x<<24) & 0xff000000) 

b = B()
uint32_swap1 = lambda x:((x>>24) & b.x ) | ((x<<8) & b.z )     | ((x>>8) & b.y   ) | ((x<<24) & b.w )


nn = (0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xdeadbeef,0xaabbccdd,0xddccbbaa)

for n in nn:
    r = R(n)
    uint32_swap2 = lambda x:( r.x & b.x ) | ( r.z & b.z ) | ( r.y & b.y ) | ( r.w & b.w )

    assert uint32_swap0(n) == uint32_swap1(n) == uint32_swap2(n) 

    print "0x%-0.8x --> 0x%-0.8x " % ( n, uint32_swap2(n)) 


