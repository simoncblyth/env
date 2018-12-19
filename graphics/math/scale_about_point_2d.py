#!/usr/bin/env python
"""



"""
from sympy import Matrix, Symbol

s = Symbol('s')
tx = Symbol('tx')
ty = Symbol('ty')

S = Matrix( ([s,0,0],[0,s,0],[0,0,1] ))

T1 = Matrix( ([1,0,0],[0,1,0],[tx,ty,1] ))

T2 = Matrix( ([1,0,0],[0,1,0],[-tx,-ty,1] ))

A = T2*S*T1

B = T1*S*T2 

print(A)
print(B)


x = Symbol('x')
y = Symbol('y')

V = Matrix( [x,y,1] )

print(A*V)

