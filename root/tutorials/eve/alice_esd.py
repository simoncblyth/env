import ROOT
import MultiView
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

## Forward declarations.

class AliESDEvent:pass
class AliESDfriend:pass
class AliESDtrack:pass
class AliExternalTrackParam:pass

def alice_esd_loadlib(file, project):pass
def make_gui():pass
def load_event():pass

def alice_esd_read():pass
def esd_make_track(trkProp, index, at,  tp=0):pass
def trackIsOn(t, mask):pass
def trackGetPos(tp):pass
def trackGetMomentum(tp):pass
def trackGetP(tp):pass

## Configuration and global variables.

esd_file_name         = "http://root.cern.ch/files/alice_ESDs.root"
esd_friends_file_name = "http://root.cern.ch/files/AliESDfriends.root"
esd_geom_file_name    = "http://root.cern.ch/files/alice_ESDgeometry.root"

esd_file          = 0
esd_friends_file  = 0
esd_tree          = 0
esd               = 0
esd_friend        = 0
esd_event_id      = 0 ## Current event id.
track_list        = 0


from ROOT import kTRUE, kFALSE, gEve


class BranchHack(dict):
    """   
        Workaround failure of TList.GetObjectRef from pyroot
        using templated code generation to allocate objects 
        for each branch of the tree from the pyside
        allowing AddressOf to be used
  
        See tutorials/tree/tcl.C for TClonesArray details
    """
    tmpl = """
global %(varname)s 
%(varname)s = ROOT.%(clsname)s(%(clcls)s)
%(treename)s.SetBranchAddress( "%(branch)s" , ROOT.AddressOf( %(varname)s ))"""
    def __init__(self, **kwa ):self.update(kwa)
    def __call__(self, el, **kwa):
        self.update(kwa)
        clsname = el.__class__.__name__ 
        self.update( {'branch':el.GetName(), 'varname':"g_%s" % el.GetName() , 'clsname':clsname , 'clcls':'' } )
        if clsname == 'TClonesArray':
            clcls = '"%s"' % el.GetClass().GetName()
            self.update( {'clcls':clcls } )

    def __repr__(self):return self.__class__.tmpl % self 


## Initialization and steering functions

def alice_esd():

    from ROOT import TFile, Error
    TFile.SetCacheFileDir(".")

    if not(alice_esd_loadlib(esd_file_name, "aliesd")):
      Error("alice_esd", "Can not load project libraries.")
      return

    print "*** Opening ESD ***"
    esd_file = TFile.Open(esd_file_name, "CACHEREAD")
    if not(esd_file):
        return

    print "*** Opening ESD-friends ***"
    esd_friends_file = TFile.Open(esd_friends_file_name, "CACHEREAD")
    if not(esd_friends_file):
        return

    global esd_tree
    esd_tree = esd_file.Get("esdTree")
    esd_tree.GetBranch("ESDfriend.").SetFile(esd_friends_file)

    global esd   
    esd = esd_tree.GetUserInfo().FindObject("AliESDEvent")

    bh = BranchHack(treename='esd_tree')
    for el in esd.fESDObjects:
        friend = el.GetName() == "AliESDfriend"
        bh(el)
        if friend:bh.update({'branch':"ESDfriend."})
        print str(bh)
        exec(str(bh))

    geom = TFile.Open(esd_geom_file_name, "CACHEREAD")
    if not(geom):
        return
    gse = geom.Get("Gentle")

    from ROOT import gEve
    gsre = ROOT.TEveGeoShape.ImportShapeExtract(gse, 0)
    geom.Close()
    del geom
    gEve.AddGlobalElement(gsre)
	
    global gMultiView
    gMultiView = MultiView.MultiView()

    gMultiView.ImportGeom(gsre)
    
    #make_gui()
    load_event()
    gEve.Redraw3D(kTRUE)  ## Reset camera after the first event has been shown.



def alice_esd_loadlib(filepath, project):
    from ROOT import gSystem, EAccessMode, TFile
    lib = "%s/%s.%s" % ( project, project, gSystem.GetSoExt() )
    if gSystem.AccessPathName(lib, EAccessMode.kReadPermission):
        f = TFile.Open(filepath, "CACHEREAD")
        if not(f):
            return kFALSE
        f.MakeProject(project, "*", "++")
        f.Close()
    return gSystem.Load(lib) >= 0


def load_event():
    global esd_tree, esd_event_id, track_list
    print "Loading event %d." %  esd_event_id
    if track_list:
        track_list.DestroyElements()

    esd_tree.GetEntry(esd_event_id)
    alice_esd_read()

    top = gEve.GetCurrentEvent()

    gMultiView.DestroyEvent()
    gMultiView.ImportEvent(top)

    gEve.Redraw3D(kFALSE, kTRUE)


## ______________________________________________________________________________
def make_gui():

    from ROOT import gEve, TRootBrowser, TGMainFrame, gClient, kDeepCleanup, TGHorizontalFrame
    browser = gEve.GetBrowser()
    browser.StartEmbedding(TRootBrowser.kLeft)

    frmMain = TGMainFrame(gClient.GetRoot(), 1000, 600)
    frmMain.SetWindowName("XX GUI")
    frmMain.SetCleanup(kDeepCleanup)

    hf = TGHorizontalFrame(frmMain)
   
    icondir = "%s/icons/" % gSystem.Getenv("ROOTSYS")  

    from ROOT import TGPictureButton

    fh = None ## EvNavHandler

    b = TGPictureButton(hf, gClient.GetPicture(icondir + "GoBack.gif"))
    hf.AddFrame(b)
    b.Connect("Clicked()", "EvNavHandler", fh, "Bck()")
    b = TGPictureButton(hf, gClient.GetPicture(icondir + "GoForward.gif"))
    hf.AddFrame(b)
    b.Connect("Clicked()", "EvNavHandler", fh, "Fwd()")

    frmMain.AddFrame(hf)
    frmMain.MapSubwindows()
    frmMain.Resize()
    frmMain.MapWindow()

    browser.StopEmbedding()
    browser.SetTabTitle("Event Control", 0)





## ______________________________________________________________________________
def alice_esd_read():

    ## Read tracks and associated clusters from current event.

    global g_AliESDRun, g_Tracks, g_AliESDfriend 
    esdrun = g_AliESDRun
    tracks = g_Tracks

    frnd   = g_AliESDfriend
    print "Friend %s, n_tracks:%d\n" % ( frnd, frnd.fTracks.GetEntries() )

    global track_list
    if track_list == 0:
        track_list = ROOT.TEveTrackList("ESD Tracks") 
        track_list.SetMainColor(6)
        track_list.SetMarkerColor(ROOT.kYellow)
        track_list.SetMarkerStyle(4)
        track_list.SetMarkerSize(0.5)
        gEve.AddElement(track_list)

    trkProp = track_list.GetPropagator()
    trkProp.SetMagField( 0.1 * esdrun.fMagneticField ) 

    kITSrefit = 4
    for n, at in enumerate(tracks):
        tp = at
        if not(trackIsOn(at, kITSrefit)):
            tp = at.fIp
     
        track = esd_make_track(trkProp, n, at, tp)
        track.SetAttLineAttMarker(track_list)
        gEve.AddElement(track, track_list)

    track_list.MakeTracks()



## ______________________________________________________________________________
def esd_make_track(trkProp, index, at, tp):
    from ROOT import TEveRecTrack
    if tp == 0:tp = at
    rt = TEveRecTrack()
    rt.fLabel  = at.fLabel
    rt.fIndex  = index
    #rt.fStatus = at.fFlags   getting : "OverflowError: long int too large to convert to int "
    if tp.fP[4] > 0:
        rt.fSign = 1 
    else:
        rt.fSign = -1

    rt.fV.Set( *trackGetPos(tp) ) 
    rt.fP.Set( *trackGetMomentum(tp) )

    ep = trackGetP(at)
    mc = 0.138              ## // at.GetMass() - Complicated funciton, requiring PID.

    from ROOT import TMath
    rt.fBeta = ep/TMath.Sqrt(ep*ep + mc*mc)
 
    track = ROOT.TEveTrack( rt, trkProp)
    track.SetName( "TEveTrack %d" % rt.fIndex )
    track.SetStdTitle()
    return track


## ______________________________________________________________________________
def trackIsOn( t, mask):
   return (t.fFlags & mask) > 0

## ______________________________________________________________________________
def trackGetPos( tp ):
    r = [ tp.fX , tp.fP[0] , tp.fP[1] ]
    cs=ROOT.TMath.Cos(tp.fAlpha)
    sn=ROOT.TMath.Sin(tp.fAlpha)
    x=tp.fX
    r[0] = x*cs - r[1]*sn 
    r[1] = x*sn + r[1]*cs
    return r

## ______________________________________________________________________________
def trackGetMomentum( tp ):
    p = [ tp.fP[4] , tp.fP[2] , tp.fP[3] ]
    pt=1./ROOT.TMath.Abs(p[0])
    cs=ROOT.TMath.Cos(tp.fAlpha)
    sn=ROOT.TMath.Sin(tp.fAlpha)
    r=ROOT.TMath.Sqrt(1 - p[1]*p[1])
    p[0]=pt*(r*cs - p[1]*sn) 
    p[1]=pt*(p[1]*cs + r*sn) 
    p[2]=pt*p[2]
    return p

## ______________________________________________________________________________
def trackGetP( tp ):
    return ROOT.TMath.Sqrt(1.+ tp.fP[3]*tp.fP[3])/ROOT.TMath.Abs(tp.fP[4])


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    alice_esd()
