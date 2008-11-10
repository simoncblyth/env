
__all__ = ['Configure']

from GaudiPython import PyAlgorithm

class MyAlg(PyAlgorithm):
	def execute(self):
		print 'Starting customizing algorithm!'
		return True

class Configure:

	def __init__(self):
        	from GaudiPython import AppMgr
        	app = AppMgr()
        	print 'Adding customizing algorithm'
        	app.addAlgorithm(MyAlg())
        	print 'Run!! Go Fight Go!!'
		return
	pass

