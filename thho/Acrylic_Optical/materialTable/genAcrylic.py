#!/usr/bin/env python

'''
Generating a new acrylic material xml file fed to GiGa.

Usage:
    shell prompt> .genAcrylic.py

    together with the function name defined in var, model

'''


import xml.etree.ElementTree as ET

# decide the file format
pageFormat = ".xml"
model = "ModelA"

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
    return absNum


def GenRefNum(model):
    if(model=="ModelA"):
        refNum=ModelRefA()
    return refNum


def ModelAbsA():
    absNum=""
    for i in range(10):
        absNum = absNum + str(i) + " "
    return absNum


def ModelRefA():
    refNum=""
    for i in range(10):
        refNum = refNum + str(i) + " "
    return refNum

if '__main__' == __name__:
    print __doc__
    GenTable(model)
    pass
