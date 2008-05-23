#
# http://www.somethingaboutorange.com/mrl/projects/nose/doc/selector_plugin.html
# http://www.somethingaboutorange.com/mrl/projects/nose/doc/plugin_interface.html
#

import sys

from nose.selector import Selector
from nose.plugins import Plugin
from nose.plugins.plugintest import run_buffered as run
from nose.util import ls_tree

class PassSelector(Selector):
	def wantDirectory(self,dirname):
		want = Selector.wantDirectory(self,dirname)
		if not(want):print "dir:%s[%s]"%(dirname, want)
		return True
	def wantFile(self,filename):
		want = Selector.wantFile(self,filename)
		print "file:%s[%s]"%(filename, want)
		return want
	def wantModule(self,module):
		want = Selector.wantModule(self,module)
		if not(want): print "module:%s[%s]"%(module, want)
		return want
	def wantClass(self,cls):
		want = Selector.wantClass(self,cls)
		if not(want): print "cls:%s[%s]"%(cls, want)
		return want
	def wantMethod(self,method):
		want = Selector.wantMethod(self,method)
		if want: print "method:%s[%s]"%(method, want )
		return False
	def wantFunction(self,function):
		want = Selector.wantFunction(self,function)
		if want: print "function:%s[%s]"%(function, want)
		return False
	
class Pass(Plugin):
	""" this allows the tests that would be run to be listed without actually running them 
	 
	    Usage
	           python $ENV_HOME/unittest/pass.py etc...  
	    also try:
	           nosetests --debug=nose.selector 
	  
	    to attempt to understand the test selection 
	"""
	enabled = True
	def configure(self, options, conf):
		pass # always on
	def prepareTestLoader(self, loader):
		loader.selector = PassSelector(loader.config)
		
if __name__=='__main__':
	#dir = sys.argv[1]
	#print ls_tree(dir)
	#argv = [__file__, '-v', dir]
	run(argv=sys.argv,  plugins=[Pass()])