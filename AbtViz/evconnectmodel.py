import ROOT
import os

class EvConnectModel:
    """
         Loads lib and instantiates the C++ EvModel 
         and provides facility to hookup signals emitted from the model 
         to callables : usually methods 
    """
    def __init__(self):
        self.load()

    def load(self):
        if ROOT.gSystem.Load("libAbtViz") < 0:ROOT.gSystem.Exit(10) 
        ROOT.EvModel.Create()
        self.instance = ROOT.EvModel.g_
        assert self.instance , self
        #print self

    def __call__(self, *args ):self.connect(*args)
    def connect( self , sign , method  ):   
        """
           Arguments :
               sign      - the string signature of the signal emitted from the EvModel
               method    - callable to be invoked when signal is emitted

        """
        handlerName = "_%s" % method.__name__
        setattr( self , handlerName , ROOT.TPyDispatcher( method ) )
        self.instance.Connect( sign , "TPyDispatcher", getattr( self , handlerName )  , "Dispatch()" ) 

    def __repr__(self):
        return repr(self.instance)



if __name__=='__main__':
    from evconnectmodel import EvConnectModel 
    ecm = EvConnectModel()


