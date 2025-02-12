#!/usr/bin/env python

def stride_repeat_0(a, stride, repeat):
    o = []
   
    it = len(a)/stride
    for item in range(0,it):
        for r in range(0,repeat):
            for offset in range(0,stride):
                j = item*stride + offset                 
                o.append(a[j]) 
    return o


def stride_repeat_1(a, stride, repeat):
    o = []

    sr = stride*repeat 
    it = len(a)/stride 
    n = sr*it

    for _ in range(0,n):
        j = stride*(_/sr) + (_ % stride)
        o.append(a[j]) 
    pass
    return o

 
def repeat_0(a, repeat):
    """
    """
    o = []
    for i in range(0,len(a)):
        for r in range(0,repeat):
            o.append(a[i]) 
    return o

 
def repeat_1(a, repeat):
    """Unnest the repeat loop"""
    o = []
    n = len(a)*repeat
    for _ in range(0,n):
        o.append( a[_/repeat])    
    return o
 

def stride_0(a, stride, offset):
    o = []
    n = len(a)/stride
    for _ in range(0,n):
        o.append( stride*a[_] + offset)
    return o
    

if __name__ == '__main__':
    a = [0,1,2,3]
    s20 = [0,2]
    s21 = [1,3]
    sr23 = [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3]
    r2 = [0,0,1,1,2,2,3,3] 


    assert stride_repeat_0(a, 2,3) == sr23
    assert stride_repeat_1(a, 2,3) == sr23 
    assert repeat_0(a,2) == r2
    assert repeat_1(a,2) == r2
    assert stride_0(a,2,0) == s20
    assert stride_0(a,2,1) == s21



