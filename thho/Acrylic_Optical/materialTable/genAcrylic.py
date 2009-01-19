#!/usr/bin/env python

'''
Generating a new acrylic material xml file fed to GiGa.

Usage:
    shell prompt> .genAcrylic.py

    together with the function name defined in var, model

'''

import xml.etree.ElementTree as ET
import math as ma
import decimal as dm


# decide the file format
pageFormat = ".xml"
model = "ModelB"

def GenTable(model):

    #tree = ET.parse("acrylic.xml")

    root = ET.Element("DDDB")
    catalogTab = ET.SubElement(root, "catalog", \
name="AcrylicProperties")
    absTab = ET.SubElement(catalogTab, "tabproperty", \
name="AcrylicAbsorptionLength",\
type="ABSLENGTH",\
xunit="eV",\
yunit="mm",\
xaxis="PhotonEnergy",\
yaxis="AbsorptionLength")

    absNum = GenAbsNum(model)
    absTab.text = absNum

    refTab = ET.SubElement(catalogTab, "tabproperty", \
name="AcrylicRefractionIndex",\
type="RINDEX",\
xunit="eV",\
yunit="",\
xaxis="PhotonEnergy",\
yaxis="IndexOfRefraction")

    refNum = GenRefNum(model)
    refTab.text = refNum

    outputTree = ET.ElementTree(root)
    outputTree.write("fake_acrylic.xml")
    pass


def GenAbsNum(model):
    if(model=="ModelA"):
        absNum = ModelAbsA()
    elif(model=="ModelB"):
        absNum = ModelAbsB(0.2,200.,3.,300.)
    else:
        print "Please specify a correct model name"
    return absNum


def GenRefNum(model):
    if(model=="ModelA"):
        refNum = ModelRefA()
    elif(model=="ModelB"):
        refNum = ModelRefB()
    else:
        print "Please specify a correct model name"
    return refNum


#################################################################
# simple output string to check the writting format
def ModelAbsA():
    absNum=""
    for i in range(1,11):
        if i%2 == 0:
            absNum = absNum + str(i) + "\n"
        else: absNum = absNum + str(i) + " "
    return absNum


def ModelRefA():
    refNum=""
    for i in range(1,11):
        if i%2 == 0:
            refNum = refNum + str(i) + "\n"
        else: refNum = refNum + str(i) + " "
    return refNum
#################################################################

#################################################################
# ModelAbs using the formula 2 in DocDB2570v3
def ModelAbsB(a1,a2,delta,cutting):
    absNum=""
    for wl in range(200,801):
        pev = float(1200)/float(wl)
        l = AbsEq(a1,a2,delta,wl,cutting)
        absNum = absNum + str(l) + " " + str(pev) +"\n"
    print absNum

    return absNum

# preseving the original G4dyb one....not complete
def ModelRefB():
    return ModelRefA()
#################################################################

#################################################################
# using the formula 2 in DocDB2570v3
#                                   (A1-A2)
# absorption length =  ------------------------------------  + A2
#                       1+exp((lambda-lambda_cutting)/delta)
def AbsEq(a1,a2,delta,wl,cutting):
    x = float(a1)-float(a2)
    y = ma.exp((float(wl)-float(cutting))/float(delta))
    return x/(float(1)+y)+a2
#################################################################

if '__main__' == __name__:
    print __doc__
    GenTable(model)
    pass
