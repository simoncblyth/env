#!/usr/bin/env python
"""
"""
import time

seconds = {}

def timing(func):
    def wrapper(*arg,**kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        global seconds
        seconds[func.func_name] = (t2-t1)
        return res 
    return wrapper


