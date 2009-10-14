
try:
    import IPython
    debug_here = IPython.Debugger.Tracer()
except ValueError:
    debug_here = lambda : None 
except AttributeError:
    debug_here = lambda : None 

