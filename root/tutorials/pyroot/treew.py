import ROOT
"""
   http://root.cern.ch/phpBB2/viewtopic.php?t=7151   

"""
def treew():
    f = ROOT.TFile("test.root","recreate")
    ntuple = ROOT.TNtuple("ntuple","Demo","px:py:pz:random:i")
    px,py,pz = [ROOT.Double(x) for x in (0,0,0)]
    for i in range(100000000):
        ROOT.gRandom.Rannor( px , py  )
        pz = px*px + py*py
        random = ROOT.gRandom.Rndm(1)
        print px,py,pz
        ntuple.Fill(px,py,pz,random,i)
        if i%1000 == 1: ntuple.AutoSave("SaveSelf")

if __name__=='__main__':
    treew()    

