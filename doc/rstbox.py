#!/usr/bin/env python
"""
rstbox.py : Code presenter by putting in RST box
==================================================

~/env/doc/rstbox.py ~/env/mandelbrot/mandelbrot.sh
~/env/doc/rstbox.py ~/env/mandelbrot/mandelbrot.cc



"""
import sys


class RSTBOX(object):
    def escapes(self, src):
        psrc = [] 
        for line in src:
            if "//S" in line: continue
            if len(line) > 0 and line[0] == "#":
                line = "&#35;" + line[1:]
            pass
            if "<" in line:line = line.replace("<","&lt;")
            if ">" in line:line = line.replace(">","&gt;")
            psrc.append(line)              
        pass
        return psrc

    def prewrap(self, src):
        psrc = [] 
        psrc.append(".. raw:: html")
        psrc.append("")
        psrc.append("   <pre class=\"mypretiny\">")
        for line in src:
            psrc.append("   " + line )     
        pass
        psrc.append("   </pre>")
        return psrc 

    def rstbox(self, src):
        ss = list(map(len, src))
        mxs = max(ss)
        horizontal_ = lambda n,c:"+-" + c * n + "-+"  
        spacer_ = lambda n,c:"| " + c * n + " |"  
        psrc = []
        psrc.append(horizontal_(mxs,"-"))
        psrc.append(spacer_(mxs," "))
        for s in src:
            pad = " " * ( mxs - len(s) ) 
            d = "| " + s + pad + " |"  
            psrc.append(d)
        pass
        psrc.append(spacer_(mxs," "))
        psrc.append(horizontal_(mxs,"-"))
        return psrc 

    def __init__(self, path):
        src = open(path).read().splitlines()
        self.src = src
        src = self.escapes(src)
        src = self.prewrap(src)
        src = self.rstbox(src)
        self.dst = src

    def __str__(self):
        return "\n".join(self.dst)



if __name__ == '__main__':
    r = RSTBOX(sys.argv[1])
    print(r)
    



