#!/usr/bin/env python2.6
"""
svg.charts.plot example
=========================

Quick and dirty convert numbers from logfiles into SVG plot

See source for meanings of the options

* /usr/local/env/plot/svg.charts-2.0.9/svg/charts/plot.py
* http://localhost/svgcharts/seconds.svg

For usage see *cqpack-*

"""
import os, sys, logging
log = logging.getLogger(__name__)
from svg.charts.plot import Plot

def read_dlist(path):
    log.info("reading from %s " % path )
    return map(eval,open(path,"r").readlines() )

def write_svg(g, path):
    log.info("writing to %s " % path )
    res = g.burn()
    with open(path,'w') as f:
        f.write(res)

def make_series( dl, xk, yks , yfmt="%5.2f"):
    m = {}
    for key in yks:
        xys = map(lambda d:(int(d[xk]),float(yfmt % float(d[key]))), dl)
        m[key] = list(sum(xys, ()))

    a = 0    
    for k,v in m.items():
        s = sum(v)
        a += s
        log.info("subtotal %s : %s   %s  " % (k, s, s/60./60.))
    pass    
    log.info(" TOTAL %s : %s   %s  " % ("all", a, a/60./60.))
    return m 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1]
    assert os.path.exists(path), path
    root,ext  = os.path.splitext(path)
    svgpath = root + '.svg'
    dl = read_dlist(path)
    
    xk  = 'index' 
    yks = 'propagate check readsrc'.split() 

    m = make_series(dl, xk, yks )
    opts = {
        'min_x_value': 0,
        'min_y_value': 0,
        'area_fill': True,
        'stagger_x_labels': False,
        'stagger_y_labels': False,
        'show_x_guidelines': False, 
        'show_y_guidelines': False, 
        'show_data_values': False, 
        'show_data_points': False, 
        'scale_x_integers': True, 
        'scale_y_integers': True, 
       }
    g = Plot( opts )
    for k in yks:   
        g.add_data({'data': m[k], 'title': k })
    pass
    write_svg(g,svgpath)

