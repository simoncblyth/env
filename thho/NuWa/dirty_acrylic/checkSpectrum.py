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
   
    c1 = TCanvas("c1DrawAllKE", "c1DrawAllKE", 800, 600)
    c1.Divide(3,1)
   
    c1.cd(1)
    strDraw = 'genKE'
    strRange = ''

    strSelect = genTypeSelect['K40']
    t.Draw(strDraw + ">>hAllKEK40" + strRange, strSelect)
    hAllKEK40 = gDirectory.FindObject("hAllKEK40")
    SetStyle(hAllKEK40, 
             title='K40', 
             xTitle='E [MeV]',
             yTitle=' ',
             )
    hAllKEK40.Draw()
    gPad.SetLogy()  
   
    c1.cd(2)
    strDraw = 'genKE'
    strRange = ''

    strSelect = genTypeSelect['U238']
    t.Draw(strDraw + ">>hAllKEU238" + strRange, strSelect)
    hAllKEU238 = gDirectory.FindObject("hAllKEU238")
    SetStyle(hAllKEU238, 
             title='U238 Gen Spectrum', 
             xTitle='E [MeV]',
             yTitle=' ',
             )
    hAllKEU238.Draw()
    gPad.SetLogy()  

    c1.cd(3)
    strDraw = 'genKE'
    strRange = ''

    strSelect = genTypeSelect['Th232']
    t.Draw(strDraw + ">>hAllKETh232" + strRange, strSelect)
    hAllKETh232 = gDirectory.FindObject("hAllKETh232")
    SetStyle(hAllKETh232,
             title='Th232 Gen Spectrum',
             xTitle='E [MeV]',
             yTitle=' ',
             )
    hAllKETh232.Draw()
    gPad.SetLogy()


    #c1.SaveAs('figs_GenDecay/allKE.png')
    return c1

# ===============================================
def DrawHitSumSpectrum(t):
    'Draw All KEs from the Decay Chain'

    c1 = TCanvas("cHitSpectrum", "cHitSpectrum", 800, 600)
    c1.Divide(3,1)

    c1.cd(1)
    strDraw = 'hitSum'
    strRange = ''

    strSelect = genTypeSelect['K40']
    t.Draw(strDraw + ">>hHitSumSpectrumK40" + strRange, strSelect)
    hHitSumSpectrumK40 = gDirectory.FindObject("hHitSumSpectrumK40")
    SetStyle(hHitSumSpectrumK40,
             title='K40 Energy Spectrum',
             xTitle='Hit Sum',
             yTitle=' ',
             )
    hHitSumSpectrumK40.Draw()
    gPad.SetLogy()

    c1.cd(2)
    strDraw = 'hitSum'
    strRange = ''

    strSelect = genTypeSelect['U238']
    t.Draw(strDraw + ">>hHitSumSpectrumU238" + strRange, strSelect)
    hHitSumSpectrumU238 = gDirectory.FindObject("hHitSumSpectrumU238")
    SetStyle(hHitSumSpectrumU238,
             title='U238 Energy Spectrum',
             xTitle='Hit Sum',
             yTitle=' ',
             )
    hHitSumSpectrumU238.Draw()
    gPad.SetLogy()

    c1.cd(3)
    strDraw = 'hitSum'
    strRange = ''

    strSelect = genTypeSelect['Th232']
    t.Draw(strDraw + ">>hHitSumSpectrumTh232" + strRange, strSelect)
    hHitSumSpectrumTh232 = gDirectory.FindObject("hHitSumSpectrumTh232")
    SetStyle(hHitSumSpectrumTh232,
             title='Th232 Energy Spectrum',
             xTitle='Hit Sum',
             yTitle=' ',
             )
    hHitSumSpectrumTh232.Draw()
    gPad.SetLogy()



    return c1


# ===============================================
def DrawADCSumSpectrum(t):
    'Draw All KEs from the Decay Chain'

    c1 = TCanvas("cADCSpectrum", "cADCSpectrum", 800, 600)
    c1.Divide(3,1)

    c1.cd(1)
    strDraw = 'adcSum'
    strRange = ''

    strSelect = genTypeSelect['K40']
    t.Draw(strDraw + ">>hADCSumSpectrumK40" + strRange, strSelect)
    hADCSumSpectrumK40 = gDirectory.FindObject("hADCSumSpectrumK40")
    SetStyle(hADCSumSpectrumK40,
             title='K40 Energy Spectrum',
             xTitle='ADC Sum',
             yTitle=' ',
             )
    hADCSumSpectrumK40.Draw()
    gPad.SetLogy()

    c1.cd(2)
    strDraw = 'adcSum'
    strRange = ''

    strSelect = genTypeSelect['U238']
    t.Draw(strDraw + ">>hADCSumSpectrumU238" + strRange, strSelect)
    hADCSumSpectrumU238 = gDirectory.FindObject("hADCSumSpectrumU238")
    SetStyle(hADCSumSpectrumU238,
             title='U238 Energy Spectrum',
             xTitle='ADC Sum',
             yTitle=' ',
             )
    hADCSumSpectrumU238.Draw()
    gPad.SetLogy()

    c1.cd(3)
    strDraw = 'adcSum'
    strRange = ''

    strSelect = genTypeSelect['Th232']
    t.Draw(strDraw + ">>hADCSumSpectrumTh232" + strRange, strSelect)
    hADCSumSpectrumTh232 = gDirectory.FindObject("hADCSumSpectrumTh232")
    SetStyle(hADCSumSpectrumTh232,
             title='Th232 Energy Spectrum',
             xTitle='ADC Sum',
             yTitle=' ',
             )
    hADCSumSpectrumTh232.Draw()
    gPad.SetLogy()

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
c3 = DrawADCSumSpectrum(t)

raw_input("press any key to continue ...")
