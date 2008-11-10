
from GaudiPython import PyAlgorithm

class MyAlg(PyAlgorithm):
	def execute(self):
		print 'Starting customizing algorithm!'
		return True



def run(*args):
	pass

def configure():
	from GaudiPython import AppMgr
	app = AppMgr()
	print 'Adding customizing algorithm'
	app.addAlgorithm(MyAlg())
	print 'Run!! Go Fight Go!!'
	return
pass

