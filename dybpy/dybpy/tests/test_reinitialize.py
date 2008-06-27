


def test_reinitialize():

    from GaudiPython import AppMgr 
    g = AppMgr()

    g.initialize()
    g.run(1)

    g.reinitialize()
    g.run(1)

    g.exit()
