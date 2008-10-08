from GaudiPython import PyAlgorithm

class MyAlg(PyAlgorithm):
    def execute(self):
        print "hello"
        return True

if __name__=="__main__":
    import gentools
    from GaudiPython import AppMgr
    app = AppMgr()
    app.addAlgorithm(MyAlg())
    app.run(10)



