#!/usr/bin/env python

'''
Analyzing the xml files contaning the acrylic sample measurements and
creating a html page to present the data in the xml files.

The xml data files are composed of: samples.xml, pieces.xml, measurements.xml

Rountine example:
    1. Edit the info according to tags in samples.xml, pieces.xml or measurements.xml.
        If these is no tag you need, add new one and edit this info on the wiki page.
        It YOUR duty to make sure there is no duplicate tag in the xml files.
    2. Decide what are the extra info tags you would like to show. Add functions like:
        ......
        GetSampleXmlTextByTag(row,sample,"author")
        ......
        GetPieceXmlTextByTag(row,piece,"history")
        ......
        GetMeasurementXmlTextByTag(row,measurement,"people")
        ......
    3. Rerun this script, then it will generate the html pages containing info table
        indicating where are the row data files and other info, e.g. an image or a histogram.



'''

#
# TODO: 1.check the file existing
#       2.check the id consistence
#       3.add general function could add xml texts by tags in the table
#       4.add duplicate Id checking function in another py script. import it
#       5.add __main__ and doc


import xml.etree.ElementTree as ET

# decide the file format
pageFormat = ".html"
rowDataFormat = ".asc"
imageFormat = ".png"

def MakeIndexHtmlPrototype(root):
    head =  ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    body =  ET.SubElement(root, "body")

    sampleTable = ET.SubElement(body, "table")
    sampleTableRowTitle = ET.SubElement(sampleTable,"tr")
    ET.SubElement(sampleTableRowTitle,"th").text = "sample chunk"
    ET.SubElement(sampleTableRowTitle,"th").text = "sample piece"

    title.text = "Dayabay acrylic transmittance measurement - sample chunks"
    sampleTable.set("border","1")

    return sampleTable
    pass

def MakePieceHtmlPrototype(root):
    head =  ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    body =  ET.SubElement(root, "body")

    sampleTable = ET.SubElement(body, "table")
    sampleTableRowTitle = ET.SubElement(sampleTable,"tr")
    ET.SubElement(sampleTableRowTitle,"th").text = "sample piece"
    ET.SubElement(sampleTableRowTitle,"th").text = "treatment and measurement"

    title.text = "Dayabay acrylic transmittance measurement - sample pieces of a chunk"
    sampleTable.set("border","1")

    sampleTable
    return sampleTable
    pass

def MakeMeasurementHtmlPrototype(root):
    head =  ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    body =  ET.SubElement(root, "body")

    sampleTable = ET.SubElement(body, "table")
    sampleTableRowTitle = ET.SubElement(sampleTable,"tr")
    ET.SubElement(sampleTableRowTitle,"th").text = "treatment and measurement"
    ET.SubElement(sampleTableRowTitle,"th").text = "row data file"

    title.text = "Dayabay acrylic transmittance measurement - sample pieces measurement"
    sampleTable.set("border","1")

    sampleTable
    return sampleTable
    pass

# generating the samples list
def MakeSamplesPage():
    root = ET.Element("html")
    sampleTable = MakeIndexHtmlPrototype(root)

    tree = ET.parse("samples.xml")
    ti = tree.getiterator("sample")

    for sample in ti:
        row = ET.SubElement(sampleTable,"tr")
        ET.SubElement(row,"td").text = sample.tag+sample.attrib.get("sampleId")
        colB = ET.SubElement(row,"td")
        for piece in sample.getiterator("piece"):
            linkFileName = sample.tag+sample.attrib.get("sampleId") + "_" + piece.tag + piece.attrib.get("pieceId") 
            href = ET.SubElement(ET.SubElement(colB,"p"),"a")
            href.set("href",linkFileName + pageFormat)
            href.text = linkFileName
            MakePiecesPage(sample,piece,linkFileName + pageFormat)
        #######################################################################################
        # Use GetSampleXmlTextByTag function here to add more info into the presentation table#
        #                                                                                     #
        #                                                                                     #
        #GetSampleXmlTextByTag(row,sample,"author")
        #######################################################################################

    outputTree = ET.ElementTree(root)
    outputTree.write("index.html")
    pass

# generating the pieces list
def MakePiecesPage(isample,ipiece,ifile):
    root = ET.Element("html")
    sampleTable = MakePieceHtmlPrototype(root)

    tree = ET.parse("pieces.xml")
    ti = tree.getiterator("piece")

    for piece in ti:
        row = ET.SubElement(sampleTable,"tr")
        # should check the sampleId in sample.xml and piece.xml consistent,not yet complete
        if (isample.attrib.get("sampleId") == piece.attrib.get("sampleId")) and \
           (ipiece.attrib.get("pieceId") == piece.attrib.get("pieceId")):
            sample = isample
            ET.SubElement(row,"td").text = sample.tag + sample.attrib.get("sampleId") + "_" + piece.tag + piece.attrib.get("pieceId")
            colB = ET.SubElement(row,"td")
            for measurement in piece.getiterator("measurement"):
                linkFileName = sample.tag + sample.attrib.get("sampleId") + "_"\
                                + piece.tag + piece.attrib.get("pieceId") + "_"\
                                + measurement.tag + measurement.attrib.get("measurementId")
                href = ET.SubElement(ET.SubElement(colB,"p"),"a")
                href.set("href",linkFileName + pageFormat)
                href.text = linkFileName
                MakeMeasurementsPage(sample,piece,measurement,linkFileName + pageFormat)
            #######################################################################################
            # Use GetSampleXmlTextByTag function here to add more info into the presentation table#
            #                                                                                     #
            #                                                                                     #
            GetPieceXmlTextByTag(row,piece,"history")
            #######################################################################################

    outputTree = ET.ElementTree(root)
    outputTree.write(ifile)
    pass



# generating the measurement list
def MakeMeasurementsPage(isample,ipiece,imeasurement,ifile):
    root = ET.Element("html")
    sampleTable = MakeMeasurementHtmlPrototype(root)

    tree = ET.parse("measurements.xml")
    ti = tree.getiterator("measurement")

    for measurement in ti:
        row = ET.SubElement(sampleTable,"tr")
        # should check the sampleId in sample.xml and piece.xml consistent,not yet complete
        if (isample.attrib.get("sampleId") == measurement.attrib.get("sampleId")) and \
           (ipiece.attrib.get("pieceId") == measurement.attrib.get("pieceId")) and \
           (imeasurement.attrib.get("measurementId") == measurement.attrib.get("measurementId")):
            sample = isample
            piece = ipiece
            ET.SubElement(row,"td").text = sample.tag + sample.attrib.get("sampleId") + "_" \
                                            + piece.tag + piece.attrib.get("pieceId") + "_" \
                                            + measurement.tag + measurement.attrib.get("measurementId")
            colB = ET.SubElement(row,"td")
            for rowData in measurement.getiterator("rowData"):
                linkFileName = sample.attrib.get("sampleId") + piece.attrib.get("pieceId") + measurement.attrib.get("measurementId")
                href = ET.SubElement(ET.SubElement(colB,"p"),"a")
                href.set("href",linkFileName + rowDataFormat)
                href.text = linkFileName
            #######################################################################################
            # Use GetSampleXmlTextByTag function here to add more info into the presentation table#
            #                                                                                     #
            #                                                                                     #
            GetMeasurementXmlTextByTag(row,measurement,"people")
            #######################################################################################

    outputTree = ET.ElementTree(root)
    outputTree.write(ifile)
    pass

# general function to access xml data by tag and put it in the table
def GetSampleXmlTextByTag(row,sample,tag):
    ET.SubElement(row,"td").text = sample.find(tag).text
    pass
def GetPieceXmlTextByTag(row,piece,tag):
    ET.SubElement(row,"td").text = piece.find(tag).text
    pass
def GetMeasurementXmlTextByTag(row,measurement,tag):
    ET.SubElement(row,"td").text = measurement.find(tag).text
    pass


if '__main__' == __name__:
    print __doc__
    MakeSamplesPage()
    # import checkId
    # CheckId()

