#!/usr/bin/env python

'''
Generating a new acrylic material xml file fed to GiGa.

Usage:
    shell prompt> .genAcrylic.py

    together with the function name defined in var, model

'''

import xml.etree.ElementTree as ET
import math as ma

# decide the file format
#model = "ModelA"
model = "ModelB"
#model = "ModelC"

def GenTable(model):

    print "\nUsing\t", model, " to generate the material xml"
    tree = ET.parse("acrylic.xml")
    for ele in tree.getiterator("tabproperty"):
        if(ele.attrib["name"]=="AcrylicAbsorptionLength"):
            DumpFindingTab(ele)
            ele.text=GenAbsNum(model)
        elif(ele.attrib["name"]=="AcrylicRefractionIndex"):
            DumpFindingTab(ele)
            ele.text=GenRefNum(model)
        else: print "Ooops! More Property??????"

    # using a "cheatting" way to deal with the doctype issues
    filename = model + "_acrylic.xml"
    f = open(filename,"w")
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd">\n')
    f.write(ET.tostring(tree.getroot()))
    f.close()
    
    print "\nDone! Output file name is\t" ,filename, "\n"
    pass


def GenAbsNum(model):
    if(model=="ModelA"):
        absNum = ModelAbsA(730.281, 2185.28, 374.751, 6.92802, 270.0)
    elif(model=="ModelB"):
        absNum = ModelAbsB()
    elif(model == "ModelC"):
        absNum = ModelAbsC(730.281, 2185.28, 374.751, 6.92802, 270.0)
    elif(model=="IORModelA"):
        pass
    else:
        print "Please specify a correct model name"
    return absNum


def GenRefNum(model):
    if(model=="ModelA"):
        refNum = ModelRefA()
        pass
    elif(model=="ModelB"):
        refNum = ModelRefB(1.47228, 4825.03)
        pass
    elif(model == "ModelC"):
        refNum = ModelRefC(1.47228, 4825.03)
    else:
        print "Please specify a correct model name"
    return refNum

def DumpFindingTab(ele):
    print "Parsing tab\t", ele.tag, " ,attrib\t", ele.attrib["name"]
    pass


#################################################################

# A Fermi-Dirac-like attenuation
def ModelAbsA(lowWl, upWl, cutWl, delta, tAbsWl):
    absNum=""
    for wl in range(190,801):
        pev = float(1200)/float(wl)
        l = AbsEq(wl, lowWl, upWl, cutWl, delta, tAbsWl)
        absNum = absNum + str(pev) + " " + str(l) +"\n"
    return absNum

# preserve the original G4dyb value
def ModelRefA():
    refNum='''
                        1.55                  1.4878
                     1.79505                  1.4895
                     2.10499                  1.4925
                     2.27077                  1.4946
                     2.55111                  1.4986
                     2.84498                  1.5022
                     3.06361                  1.5065
                     4.13281                  1.5358
                        6.20                  1.6279'''
    return refNum




# preserve the original G4dyb value
def ModelAbsB():
    absNum='''
                        1.55                   5.0e3
                        1.61                   5.0e3
                        2.07                   5.0e3
                        2.48                   5.0e3
                        3.76                   5.0e3
                        4.13                   5.0e3
                        6.20                  5.0e-3'''
    return absNum

# Cauchy equ
def ModelRefB(cauchyA, cauchyB):
    refNum=""
    for wl in range(190,801):
        pev = float(1200)/float(wl)
        ior = RefEq(wl, cauchyA, cauchyB)
        refNum = refNum + str(pev) + " " + str(ior) + "\n"
    return refNum




# Cauchy equ and Fermi-Dirac-like attenuation
def ModelAbsC(lowWl, upWl, cutWl, delta, tAbsWl):
    absNum=""
    for wl in range(190,801):
        pev = float(1200)/float(wl)
        l = AbsEq(wl, lowWl, upWl, cutWl, delta, tAbsWl)
        absNum = absNum + str(pev) + " " + str(l) + "\n"
    return absNum

def ModelRefC(cauchyA, cauchyB):
    refNum=""
    for wl in range(190,801):
        pev = float(1200)/float(wl)
        ior = RefEq(wl, cauchyA, cauchyB)
        refNum = refNum + str(pev) + " " + str(ior) + "\n"
    return refNum


#################################################################

#################################################################
# A Fermi-Dirac-like distribution
#                                   (lowWl-upWl)
# absorption length =  ------------------------------------  + upWl + (lambda/totalAbsWl)*lowWl
#                       1+exp((lambda-lambda_cutting)/delta)
def AbsEq(wl, lowWl, upWl, cutWl, delta, tAbsWl):
    x = float(lowWl)-float(upWl)
    y = ma.exp((float(wl)-float(cutWl))/float(delta))
    z = (float(tAbsWl)/float(wl))*float(lowWl)
    att = (x/(float(1)+y))+float(upWl)-z
    #print x, y, z, wl, att
    if( att < 0 ):
        att = 0
    return att

# Cauchy eq
def RefEq(wl, cauchyA, cauchyB):
    x = float(cauchyA)
    y = (float(cauchyB))/(float(wl)*float(wl))
    return x+y


#################################################################

if '__main__' == __name__:
    #print __doc__
    GenTable(model)
    pass
