#!/usr/bin/env python
"""
::

    python ~/env/base/timing.py
    {'cmeth': 0.10109281539916992, 'other': 0.5005331039428711, '__init__': 1.9073486328125e-06}

"""
import time, logging
log = logging.getLogger(__name__)


def timing_report(klss):
    log.info("timing_report")
    for kls in klss:
        print( "%-10s " % (kls.__name__)) 
        keys = map(lambda _:_[:-4],filter(lambda _:_[-4:] == '_tot', kls.secs.keys()))
        for k in sorted(keys):
            tot = kls.secs["%s_tot" % k]
            num = kls.secs["%s_num" % k]
            avg = kls.secs["%s_avg" % k]
            print( "%-30s : %10.3f %10d %10.3f " % (k, tot, num, avg )) 


def timing(results): 
    """
    # three levels allows the decorator to take an argument 
    """
    def real_timing(func):
        def wrapper(*arg,**kw):
            """
            As this is invoked, during the wrapping, the func
            is not yet an instance method (no im_class), 
            so adopt devious approach of passing in a class variable 
            as an argument to the decorator, in order to leave the timing 
            results in the class variable.
            """
            t0 = time.time()
            res = func(*arg,**kw)

            key = func.func_name

            tot = "%s_tot" % key
            num = "%s_num" % key
            avg = "%s_avg" % key

            if not tot in results:
                results[tot] = 0
                results[num] = 0
                results[avg] = 0
            pass
            results[tot] += time.time()-t0 
            results[num] += 1
            results[avg] = results[tot]/results[num]

            return res 
        return wrapper
    return real_timing 


if __name__ == '__main__':

    class Demo(object):
        secs = {}
        @timing(secs)
        def __init__(self):
            pass
        @timing(secs)
        def other(self):
            time.sleep(0.5)

        @classmethod
        @timing(secs)
        def cmeth(cls):
            time.sleep(0.1)
          
     
    Demo.cmeth()
    d = Demo()
    d.other()
    print Demo.secs

