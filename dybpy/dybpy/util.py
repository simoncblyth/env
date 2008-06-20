
def syspath():
    import sys
    for s in sys.path:
        print s

def gt():
    """
       dybgaudi/InstallArea/python/gentools.py
    """

    import xmldetdesc
    xddc = xmldetdesc.XmlDetDescConfig()
    tg = GenToolsConfig(volume="/dd/Geometry/Pool/lvFarPoolIWS")
    import gaudimodule as gm
    app = gm.AppMgr()
    app.EvtSel = "NONE"
    app.run(tg.nevents)


if __name__=='__main__':
    gt()








