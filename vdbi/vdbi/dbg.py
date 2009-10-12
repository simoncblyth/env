
try:
    import IPython
    debug_here = IPython.Debugger.Tracer()
except ValueError:
    debug_here = lambda : None 


