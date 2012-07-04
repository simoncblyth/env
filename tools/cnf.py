#!/usr/bin/env python
import os
from ConfigParser import RawConfigParser
RawConfigParser.optionxform = str    # case sensitive keys 

class Cnf(RawConfigParser):
    """
    Simple enhancements to RawConfigParser

    #. expand user and envvar in paths
    #. use case sensitive keys
    #. dictionary based interface

    """
    def read(self, paths ):
	if isinstance(paths, basestring):
	    paths = [paths]
	filenames = []    
        for path in paths:
	    filenames.append(os.path.expandvars(os.path.expanduser(path)))    	
        return RawConfigParser.read(self,filenames)

    def sectiondict(self, sect):
        d = {}
        if self.has_section(sect):
            for key in self.options(sect):    
	        d[key] = self.get(sect,key)
        return d

    def asdict(self):
        d = {}
	for sect in self.sections():
            d[sect] = self.sectiondict(sect)
	return d    
 
if __name__ == '__main__':
    cnf = Cnf()
    cnf.read("~/.libfab.cnf")
    print cnf.sections()
    print cnf.asdict()

