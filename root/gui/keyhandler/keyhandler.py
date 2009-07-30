"""

   This is goin nowhere ... 
       key handling needs precise signature calls / casting
       that cannot do from py side

   so move the "state" of the event display onto the compiled side
   to allow it to be manipulated from both py and c++ 


"""

import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
ROOT.gROOT.ProcessLine(".L KeyHandler.cxx+")
from ROOT import KeyHandler


def Handler():
    ## gTQSender is pointer to the object that sent the last signal
    obj = ROOT.BindObject( ROOT.gTQSender, ROOT.TEveBrowser )
    print obj


dispatcher = ROOT.TPyDispatcher( Handler )

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    from ROOT import gEve, gVirtualX

    kh = KeyHandler()
    browser = gEve.GetBrowser()
    toolbar = browser.GetToolbarFrame()

    browser.BindKey( f , gVirtualX.KeysymToKeycode(ROOT.kKey_Space) , 0 )

    ## problem here is that the target needs to have a HandleKey(Event_t ) method 

    ## ... hmm  



   

