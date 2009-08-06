
import ROOT

def treer():
  f = ROOT.TFile("test.root")
  ntuple = f.Get("ntuple")
  c1 = ROOT.TCanvas()
  first = 0
  while True:
     if first == 0: ntuple.Draw("px>>hpx", "","",10000000,first)
     else:          ntuple.Draw("px>>+hpx","","",10000000,first)
     first = ntuple.GetEntries()
     c1.Update()
     ROOT.gSystem.Sleep(1000)
     ntuple.Refresh()


if __name__=='__main__':
    treer()

