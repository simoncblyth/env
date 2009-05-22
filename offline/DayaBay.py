
def enums(e):
    d = {}
    import inspect
    for k,v in inspect.getmembers(e):
        if k.startswith('k'):d.update( {k:v} )
    return d


import pickle

class DayaBay(dict):
    name = "DayaBay.pk"
    def __init__(self):
        """
            Usage :
                from env.offline.DayaBay import DayaBay
                dyb = DayaBay()
                dyb.keys()
                ['DetectorId', 'Site']

             Transport the enums out of the nuwa environment ... 
             
             NB need to access the Detector class first 
             before the enums are conjured into existance
        """
        try:
            import GaudiPython as gp
            Detector = gp.gbl.DayaBay.Detector
            Site = gp.gbl.Site
            DetectorId = gp.gbl.DetectorId
            self['Site'] = enums(Site)
            self['DetectorId'] = enums(DetectorId)
            self.save()
        except ImportError:
            self.load()

    def path(self):
        import os
        return os.path.join( os.path.dirname(__file__) , DayaBay.name )

    def load(self):
        """
            Default protocol is 0 : which is ascii 
        """
        d = pickle.load( file(self.path(),"r") )
        self.update( d )

    def save(self):
        pickle.dump( dict(self) , file(self.path(),"w") )


if __name__=='__main__':
    dyb = DayaBay()
    print dyb



