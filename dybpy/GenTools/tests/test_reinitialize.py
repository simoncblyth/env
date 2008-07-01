


def test_reinitialize():

    from GaudiPython import AppMgr 
    g = AppMgr()

    loc = '/Event/Gen/GenHeader' 

    import genrepr
    import gentools    
    gen = g.algorithm("GenAlg")
    gen.GenTools = [ "GtGunGenTool", "GtTimeratorTool" ]

    g.initialize()
    
    
    g.reinitialize()
    g.run(1)
    ghr = g.evtsvc()[loc]
    print repr(ghr)

    g.reinitialize()
    g.run(1)
    ghr = g.evtsvc()[loc]
    print repr(ghr)


    g.exit()
