#!/usr/bin/env python
'''
save diagnostic plots for simulated files
Modified from people/zhang/scripts/simulation/check/checkGenDecay.py
'''

import ROOT
from ROOT import gROOT, gStyle, gDirectory, gPad
from ROOT import TCanvas, TLegend, TLatex, TMath
from ROOT import TChain

def SetStyle(h, *seq, **kwargs):
    'Set the Paiter Style'
   
    title = kwargs.get('title', '')
    xTitle = kwargs.get('xTitle', '')
    yTitle = kwargs.get('yTitle', '')
    lineColor = kwargs.get('lineColor', '')
    markerColor = kwargs.get('markerColor', '')
    markerSize = kwargs.get('markerSize', '')
    markerStyle = kwargs.get('markerStyle', '')
   
    h.SetLineWidth(2)
    if title: h.SetTitle(title)
    if xTitle: h.SetXTitle(xTitle)
    if yTitle: h.SetYTitle(yTitle)
    if lineColor: h.SetLineColor(lineColor)
    if markerColor: h.SetMarkerColor(markerColor)
    if markerSize: h.SetMarkerSize(markerSize)
    if markerStyle: h.SetMarkerStyle(markerStyle)

genTypeSelect = {
    "U238" : "genType>20 && genType<30",
    "Th232" : "genType>30 && genType<40",
    "K40" : "genType>40 && genType<50",
    "Co60" : "genType>50 && genType<60"
}
pdgTypes = {
    "alpha" : '1000020040',
    "electron"  : '11', "positron" : '-11',
    "gamma" : '22',
    "neutron" : '2112',
    "positron" : '-11',
    "Ca40" : ' 1000200400 ',
    "Ar40" : ' 1000180400 ',
}

# ===============================================
def DrawAllKE(t):
    'Draw All KEs from the Decay Chain'
   
    c1 = TCanvas("c1", "c1", 800, 600)
    c1.Divide(2,2)
   
    strDraw = 'genKE'
    strRange = ''

    c1.cd(1)
    strSelect = genTypeSelect['K40']
    print strSelect
    t.Draw(strDraw + ">>hAllKEK40" + strRange, strSelect)
    hAllKEK40 = gDirectory.FindObject("hAllKEK40")
    SetStyle(hAllKEK40, 
             title='K-40', 
             xTitle='E [MeV]',
             yTitle=' ',
             )
    hAllKEK40.Draw()
    gPad.SetLogy()  

    c1.cd(2)
    strSelect = 'genPDG==' + pdgTypes["electron"] + ' && ' + genTypeSelect['K40']
    print strSelect
    t.Draw(strDraw + ">>hBetaKEK40" + strRange, strSelect)
    hBetaKEK40 = gDirectory.FindObject("hBetaKEK40")
    SetStyle(hBetaKEK40, 
             title='K-40, Beta', 
             xTitle='E [MeV]',
             yTitle=' ',
             )
    hBetaKEK40.Draw()
    gPad.SetLogy()

    c1.cd(3)
    strSelect = 'genPDG==' + pdgTypes["gamma"] + ' && ' + genTypeSelect['K40']
    print strSelect
    t.Draw(strDraw + ">>hGammaKEK40" + strRange, strSelect)
    hGammaKEK40 = gDirectory.FindObject("hGammaKEK40")
    SetStyle(hGammaKEK40, 
             title='K-40, Gamma',
             xTitle='E [MeV]',
             yTitle=' ',
             )
    hGammaKEK40.Draw()
    #gPad.SetLogy()

    c1.cd(4)
    strSelect = '(genPDG!=' + pdgTypes["gamma"] +')'+ ' && '+ '(genPDG!=' + pdgTypes["electron"] +')'  + ' && ' + genTypeSelect['K40']
    t.Draw(strDraw + ">>hOtherKEK40" + strRange, strSelect)
    hOtherKEK40 = gDirectory.FindObject("hOtherKEK40")
    SetStyle(hOtherKEK40, 
             title='K-40, Other',
             xTitle='E [MeV]',
             yTitle=' ',
             )
    hOtherKEK40.Draw()
    #gPad.SetLogy()

    c1.SaveAs('figs_GenDecay/allKE_1.png')
    return c1

def DrawStable(t):

    c2 = TCanvas("c2", "c2", 800, 600)
    c2.Divide(2,2)

    strDraw = 'genKE'
    strRange = ''


    #strDraw= 'genPDG'
    c2.cd(1)
    strSelect = 'genPDG ==' + pdgTypes["Ca40"] + ' && ' + genTypeSelect['K40']
    print strSelect
    t.Draw(strDraw + ">>hCa40K40" + strRange, strSelect)
    hCa40K40 = gDirectory.FindObject("hCa40K40")
    SetStyle(hCa40K40,
             title='K-40, Ca40',
             xTitle='E [MeV]',
             yTitle=' ',
             )
    #hOtherPDGK40.Draw()
    #print hOtherPDGK40.GetBinContent(hOtherPDGK40.GetMaximumBin())
    #print hOtherPDGK40.GetBinLowEdge(hOtherPDGK40.GetMaximumBin())

    c2.cd(2)
    strSelect = 'genPDG ==' + pdgTypes["Ar40"] + ' && ' + genTypeSelect['K40']
    print strSelect
    t.Draw(strDraw + ">>hAr40K40" + strRange, strSelect)
    hAr40K40 = gDirectory.FindObject("hAr40K40")
    SetStyle(hAr40K40,
             title='K-40, Ar40',
             xTitle='E [MeV]',
             yTitle=' ',
             )


    c2.cd(3)
    #strSelect = 'genPDG ==' + pdgTypes["positron"] + ' && ' + genTypeSelect['K40']
    strSelect = 'genPDG ==' + pdgTypes["positron"]   + ' && ' + genTypeSelect['K40']
    print strSelect
    t.Draw(strDraw + ">>hPositronK40" + strRange, strSelect)
    hPositronK40 = gDirectory.FindObject("hPositronK40")
    SetStyle(hPositronK40,
             title='K-40, Positron',
             xTitle='E [MeV]',
             yTitle=' ',
             )



    c2.cd(4)
    #strSelect = 'genKE < 0.2 && genKE > 0 ' + ' && ' + genTypeSelect['K40']
    strSelect = 'genPDG == 0 ' + ' && ' + genTypeSelect['K40']
    print strSelect
    t.Draw(strDraw + ">>hUnknownK40" + strRange, strSelect)
    #t.Draw("genPDG >> hK40" + strRange, strSelect)
    hUnknownK40 = gDirectory.FindObject("hUnknownK40")
    SetStyle(hUnknownK40,
             title='K-40, ',
             xTitle='E [MeV]',
             yTitle=' ',
             )


   
    c2.SaveAs('figs_GenDecay/allKE_2.png')
    return c2


# ===============================================
def DrawKE(t, particle='alpha', genType='U238', title='', log=False):
    'Draw individual particle'
   
    c1 = TCanvas("c1", "c1", 800, 600)
   
    strDraw = 'genKE'
    strRange = ''
    strSelect = 'genPDG==' + pdgTypes[particle] \
              + ' && ' + genTypeSelect.get(genType, "")

    t.Draw(strDraw + ">>hKE" + strRange, strSelect)
    hKE = gDirectory.FindObject("hKE")
    SetStyle(hKE, 
             title=title, 
             xTitle='E [MeV]',
             yTitle=' ',
             )

    hKE.Draw()
   
    if log: gPad.SetLogy()
   
    c1.SaveAs('figs_GenDecay/'+genType+particle+'KE.png')

    return c1
   
# ===============================================
def DrawDaughterKE(t, genType='U238',
                   daughters=['92238','','',''],
                   titles=['U-238','','',''],
                   figname='daughter.png'
                   ):
    'Draw daughter particle decays'
   
    c1 = TCanvas("c1", "c1", 800, 600)
    c1.Divide(2,2)
   
    hDaugherKE = [None] * 4
   
    for i , daughter in enumerate(daughters):
        c1.cd(i+1)
        strDraw = 'genKE'
        strRange = ''
        strSelect = 'genKE > 0.01 && genParentPDG[0] == 1000' + daughter \
                  + '0 && ' + genTypeSelect.get(genType, "")
        index = str(i)
        t.Draw(strDraw + ">>hDaugherKE" + index + strRange, strSelect)
        hDaugherKE[i] = gDirectory.FindObject("hDaugherKE" + index)
        if hDaugherKE[i]:
            SetStyle(hDaugherKE[i], 
                     title=titles[i], 
                     xTitle='E [MeV]',
                     yTitle=' ',
                     )
            hDaugherKE[i].Draw()
            gPad.SetLogy()
   
    c1.SaveAs('figs_GenDecay/'+figname)
    return c1

# ===============================================
def DrawDt(t):
    'Draw dorrelated decay time difference'
   
    c1 = TCanvas("c1", "c1", 800, 600)
    c1.Divide(2,2)
   
    c1.cd(1)
    strDraw = '(genParentT[1]-genParentT[0])/1e6'
    strRange = '(200,0,20)'
    strSelect = 'nVtx>1 && (genParentT[1]-genParentT[0])>0.1 && genParentPDG[0]==1000832140 && '
    t.Draw(strDraw + ">>hDtBi214" + strRange, strSelect + genTypeSelect['U238'])
    hDtBi214 = gDirectory.FindObject("hDtBi214")
    SetStyle(hDtBi214, 
             title='^{214}Bi - ^{214}Po (#tau_{1/2} = 164 #muSec)', 
             xTitle='#DeltaT [mSec]',
             yTitle=' ',
             )
    hDtBi214.Draw()
    hDtBi214.Fit("expo","","",0.1,10)
    decaytime = -TMath.Log(2)/hDtBi214.GetFunction("expo").GetParameter(1)
    print "Po-214 decay time :", decaytime, "mSec"
    gPad.SetLogy()
   
    c1.cd(2)
    strDraw = '(genParentT[1]-genParentT[0])/1e6'
    strRange = '(200,0,20)'
    strSelect = 'nVtx>1 && (genParentT[1]-genParentT[0])>0.1 && genParentPDG[0]==1000832120 && '
    t.Draw(strDraw + ">>hDtBi212" + strRange, strSelect + genTypeSelect['Th232'])
    hDtBi212 = gDirectory.FindObject("hDtBi212")
    SetStyle(hDtBi212, 
             title='^{212}Bi - ^{212}Po (#tau_{1/2} = 299 nSec)', 
             xTitle='#DeltaT [mSec]',
             yTitle=' ',
             )
    hDtBi212.Draw()
    hDtBi212.Fit("expo","","",0.1,10)
    decaytime = -TMath.Log(2)/hDtBi212.GetFunction("expo").GetParameter(1)
    print "Po-212 decay time :", decaytime, "mSec"
    gPad.SetLogy()
   
    c1.cd(3)
    strDraw = '(genParentT[1]-genParentT[0])/1e6'
    strRange = '(200,0,20)'
    strSelect = 'nVtx>1 && (genParentT[1]-genParentT[0])>0.1 && genParentPDG[0]==1000862200 && '
    t.Draw(strDraw + ">>hDtRn220" + strRange, strSelect + genTypeSelect['Th232'])
    hDtRn220 = gDirectory.FindObject("hDtRn220")
    SetStyle(hDtRn220, 
             title='^{220}Rn - ^{216}Po (#tau_{1/2} = 145 mSec)', 
             xTitle='#DeltaT [mSec]',
             yTitle=' ',
             )
    hDtRn220.Draw()
    hDtRn220.Fit("expo","","",0.1,10)
    decaytime = -TMath.Log(2)/hDtRn220.GetFunction("expo").GetParameter(1)
    print "Po-216 decay time :", decaytime, "mSec"
    gPad.SetLogy()
   
    c1.cd(4)
    strDraw = '(genParentT[1]-genParentT[0])/1e6'
    strRange = '(200,0,20)'
    strSelect = 'nVtx>1 && (genParentT[1]-genParentT[0])>0.1 && genParentPDG[0]==1000822100 && '
    t.Draw(strDraw + ">>hDtPb210" + strRange, strSelect + genTypeSelect['U238'])
    hDtPb210 = gDirectory.FindObject("hDtPb210")
    SetStyle(hDtPb210, 
             title='^{210}Pb - ^{210}Bi (#tau_{1/2} = 5 days)', 
             xTitle='#DeltaT [mSec]',
             yTitle=' ',
             )
    hDtPb210.Draw()
    hDtPb210.Fit("expo","","",0.1,10)
    decaytime = -TMath.Log(2)/hDtPb210.GetFunction("expo").GetParameter(1)
    print "Bi-210 decay time :", decaytime, "mSec"
    gPad.SetLogy()
   
   
    c1.SaveAs('figs_GenDecay/correlatedDecay.png')
    return c1
   
# ===============================================

gROOT.Reset()
gROOT.SetStyle("Plain")
gStyle.SetOptStat('e')
gStyle.SetPalette(1)

t = TChain("stats/tree/genTree")
t.Add("out.root")

c1 = DrawAllKE(t)
c2 = DrawStable(t)
#c1 = DrawKE(t, particle='alpha', genType='U238', title="Alpha's in U-238 Chain")
# c1 = DrawKE(t, particle='alpha', genType='Th232',
#             title="Alpha's in Th-232 Chain")
# c1 = DrawKE(t, particle='gamma', genType='U238',
#             title="Gamma's in U-238 Chain")
# c1 = DrawKE(t, particle='gamma', genType='Th232',
#             title="Gamma's in Th-232 Chain")
# c1 = DrawKE(t, particle='electron', genType='U238',
#             title="Beta's in U-238 Chain", log=True)
# c1 = DrawKE(t, particle='electron', genType='Th232',
#             title="Beta's in Th-232 Chain", log=True)
#c1 = DrawKE(t, particle='electron', genType='K40', title="Beta's in K-40 Chain")
#c1 = DrawKE(t, particle='gamma', genType='K40', title="Gamma's in K-40 Chain")
#c1 = DrawDaughterKE(t, genType='U238',
#                     daughters=['92238', '90234', '91234', '92234'],
#                     titles=['U-238', 'Th-234', 'Pa-234', 'U-234'],
#                     figname='U238KE1.png')
# c1 = DrawDaughterKE(t, genType='U238',
#                     daughters=['90230', '88226', '86222', '84218'],
#                     titles=['Th-230', 'Ra-226', 'Rn-222', 'Po-218'],
#                     figname='U238KE2.png')
# c1 = DrawDaughterKE(t, genType='U238',
#                     daughters=['82214', '83214', '82210', '83210'],
#                     titles=['Pb-214', 'Bi-214', 'Pb-210', 'Bi-210'],
#                     figname='U238KE3.png')
# c1 = DrawDaughterKE(t, genType='U238',
#                     daughters=['84210'],
#                     titles=['Po-210'],
#                     figname='U238KE4.png')
# c1 = DrawDaughterKE(t, genType='Th232',
#                     daughters=['90232', '88228', '89228', '90228'],
#                     titles=['Th-232', 'Ra-228', 'Ac-228', 'Th-228'],
#                     figname='Th232KE1.png')
# c1 = DrawDaughterKE(t, genType='Th232',
#                     daughters=['88224', '86220', '82212', '83212'],
#                     titles=['Ra-224', 'Rn-220', 'Pb-212', 'Bi-212'],
#                     figname='Th232KE2.png')
# c1 = DrawDaughterKE(t, genType='Th232',
#                     daughters=['81208'],
#                     titles=['Tl-208'],
#                     figname='Th232KE3.png')
#c1 = DrawDt(t)
raw_input("press any key to continue ...")
