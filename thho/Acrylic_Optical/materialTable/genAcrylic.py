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
pageFormat = ".xml"
model = "ModelA"

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
    filename = model + "_acrylic.xml"
    tree.write(filename)

    # using a "cheatting" way to deal with the doctype issues
    input = file(filename).read()
    rootEle = unserialize(input)
    f = open(filename,"w")
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd">\n')
    f.write(ET.tostring(rootEle))
    f.close()
    
    print "\nDone! Output file name is\t" ,filename
    pass


def GenAbsNum(model):
    if(model=="ModelA"):
        absNum = ModelAbsA(0.2,200.,3.,300.)
    elif(model=="ModelB"):
        #absNum = ModelAbsB()
        pass
    else:
        print "Please specify a correct model name"
    return absNum


def GenRefNum(model):
    if(model=="ModelA"):
        refNum = ModelRefA()
    elif(model=="ModelB"):
        #refNum = ModelRefB()
        pass
    else:
        print "Please specify a correct model name"
    return refNum

def DumpFindingTab(ele):
    print "Parsing tab\t", ele.tag, " ,attrib\t", ele.attrib["name"]
    pass

def unserialize(text):
    import StringIO
    file = StringIO.StringIO(text)
    tree = ET.parse(file)
    return tree.getroot()

#################################################################
# ModelAbs using the formula 2 in DocDB2570v3
def ModelAbsA(a1,a2,delta,cutting):
    absNum=""
    for wl in range(200,801):
        pev = float(1200)/float(wl)
        l = AbsEq(a1,a2,delta,wl,cutting)
        absNum = absNum + str(l) + " " + str(pev) +"\n"
    return absNum

def ModelRefA():
    RefNum=""
    for wl in range(200,801):
        pev = float(1200)/float(wl)
        RefNum = RefNum + str(1.5) + " " + str(pev) +"\n"
    return RefNum
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
    #print __doc__
    GenTable(model)
    pass
