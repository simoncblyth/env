#!/usr/bin/env python

from ctypes import *

class Mask(Structure):
    _fields_ = [ 
         ("m0", c_uint32 , 32 ),    
         ("m1", c_uint32 , 32 ),    
         ("m2", c_uint32 , 32 ),    
         ("m3", c_uint32 , 32 ),    
         ("m4", c_uint32 , 32 ),    
         ("m5", c_uint32 , 32 ),
              ]

if __name__ == '__main__':
    print Mask.m0
    


