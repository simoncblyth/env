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
   
    c1 = TCanvas("cDrawAllKE", "cDrawAllKE", 800, 600)

    strDraw = 'genKE'
    strRange = ''

    #strSelect = genTypeSelect['U238']
    strSelect = ''
    t.Draw(strDraw + ">>hAllKE" + strRange, strSelect)
    hAllKE = gDirectory.FindObject("hAllKE")
    SetStyle(hAllKE, 
             title='', 
             xTitle='E [MeV]',
             yTitle=' ',
             )
    hAllKE.Draw()
    #gPad.SetLogy()  

    #c1.SaveAs('figs_GenDecay/allKE.png')
    return c1

# ===============================================
def DrawHitSumSpectrum(t):
    'Draw All KEs from the Decay Chain'

    c1 = TCanvas("cHitSpectrum", "cHitSpectrum", 800, 600)

    strDraw = 'hitSum'
    strRange = ''

    #strSelect = genTypeSelect['U238']
    #strSelect = ''
    #strSelect = "genType<21 || genType>29"
    #strSelect = "genKE > 1"
    #strSelect = "genPDG == %(gamma)s"%pdgTypes
    #strSelect = "genPDG == %(electron)s"%pdgTypes
    #strSelect = "(((genPDG == %(gamma)s) || (genPDG == %(electron)s) || (genPDG == %(alpha)s)) && (hitSum > 200))"%pdgTypes
    strSelect = "hitSum > 200"
    print strSelect

    t.Draw(strDraw + ">>hHitSumSpectrum" + strRange, strSelect)
    hHitSumSpectrum = gDirectory.FindObject("hHitSumSpectrum")
    SetStyle(hHitSumSpectrum,
             title=' Energy Spectrum',
             xTitle='Hit Sum',
             yTitle=' ',
             )
    hHitSumSpectrum.Draw()
    #gPad.SetLogy()


    return c1


# ===============================================
def DrawADCSumSpectrum(t):
    'Draw All KEs from the Decay Chain'

    c1 = TCanvas("cADCSpectrum", "cADCSpectrum", 800, 600)

    strDraw = 'adcSum'
    strRange = ''

    #strSelect = genTypeSelect['']
    strSelect = ''
    t.Draw(strDraw + ">>hADCSumSpectrum" + strRange, strSelect)
    hADCSumSpectrum = gDirectory.FindObject("hADCSumSpectrum")
    SetStyle(hADCSumSpectrum,
             title=' Energy Spectrum',
             xTitle='ADC Sum',
             yTitle=' ',
             )
    hADCSumSpectrum.Draw()
    gPad.SetLogy()

    return c1

# ===============================================
def DrawSpecial(t):

    c1 = TCanvas("cSpecial", "cSpecial", 800, 600)

    strDraw = 'genType'
    strRange = ''

    #strSelect = genTypeSelect['']
    strSelect = 'hitSum == 0 && (genType != 21)'
    t.Draw(strDraw + ">>hSpecial" + strRange, strSelect)
    hSpecial = gDirectory.FindObject("hSpecial")
    SetStyle(hSpecial,
             title='what?',
             xTitle='',
             yTitle=' ',
             )
    hSpecial.Draw()
    gPad.SetLogy()

    return c1

# ===============================================
def DrawGenPositionXYRZ(t):
   
    c1 = TCanvas("cDrawGenPositionXYRZ", "cDrawGenPositionXYRZ", 800, 600)
    c1.Divide(2,1)

    c1.cd(1)
    strDraw = 'genZ : sqrt(genX*genX + genY*genY)'
    strRange = ''

    #strSelect = genTypeSelect['U238']
    strSelect = ''
    t.Draw(strDraw + ">>hRZ" + strRange, strSelect)
    hRZ = gDirectory.FindObject("hRZ")
    SetStyle(hRZ, 
             title='', 
             xTitle='',
             yTitle=' ',
             )
    hRZ.Draw()


    c1.cd(2)
    strDraw = 'genX : genY'
    strRange = ''

    #strSelect = genTypeSelect['U238']
    strSelect = ''
    t.Draw(strDraw + ">>hXY" + strRange, strSelect)
    hXY = gDirectory.FindObject("hXY")
    SetStyle(hXY, 
             title='', 
             xTitle='',
             yTitle=' ',
             )
    hXY.Draw()





    #c1.SaveAs('figs_GenDecay/allKE.png')
    return c1



# ===============================================

gROOT.Reset()
gROOT.SetStyle("Plain")
gStyle.SetOptStat('e')
gStyle.SetPalette(1)

t = TChain("stats/tree/genTree")

t.Add("out.root")

c1 = DrawAllKE(t)
c2 = DrawHitSumSpectrum(t)
#c3 = DrawADCSumSpectrum(t)
#c4 = DrawSpecial(t)
#c5 = DrawGenPositionXYRZ(t)
#c6 = DrawGenPositionXYZ

raw_input("press any key to continue ...")
