from GaudiPython import PyAlgorithm

class MyAlg(PyAlgorithm):
    def execute(self):
	print 'Starting self defining algorithm'
        return True

if __name__=="__main__":
    print ''
    import mysim
    from GaudiPython import AppMgr
    app = AppMgr()
    print 'adding self defining algorithm'
    app.addAlgorithm(MyAlg())
    print 'Run!! Go fight go!!'
    app.run(1) 

