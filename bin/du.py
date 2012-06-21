#!/usr/bin/env python
"""
Usage::

    du.py 	
    du.py /some/directory  	

"""
import os, re, logging
log = logging.getLogger(__name__)

class Entry(dict):
    units = dict(B=1,K=1024,M=1024*1024,G=1024*1024*1024,T=1024*1024*1024*1024,P=1024*1024*1024*1024*1024)
    """
    Use unit suffixes: Byte, Kilobyte, Megabyte, Gigabyte, Terabyte and Petabyte.
    """
    def __init__(self, val, unit, path ):
        self['val'] = val   
        self['unit'] = unit   
        self['path'] = path
        self['valu'] = float(val)*self.units[unit]    

    def __repr__(self):
	return "%(val)s%(unit)s\t%(valu)s\t%(path)s" % self    

class Du(list):
    def __init__(self, path=os.getcwd() ):
	cmd = "du -hs %s/*" % path 
	log.info("doing %s " % cmd )
	ptn = re.compile("^\s*([\d\.]*)([BKMGTP]{1})\t(.*)$")
	for line in os.popen(cmd).readlines():
	    m = ptn.match(line)
	    if not m:
                log.info("failed to match %s " % line)    
		continue
            groups = m.groups()
	    assert len(groups) == 3, "unexpected groups %s " % repr(groups)
	    entry = Entry(*groups)
            self.append(entry)	     

    def __repr__(self):
	return "\n".join( map(repr,sorted(self, key=lambda _:_['valu'], reverse=True )) )     


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)	
    du = Du()
    print du


