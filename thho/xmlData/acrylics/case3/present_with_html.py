#!/usr/bin/env python

'''
Analyzing the xml files contaning the acrylic sample measurements and
creating a html page to present the data in the xml files.

The xml data files are composed of: samples.xml, pieces.xml, measurements.xml

Rountine example:
    1. Edit the info according to tags in samples.xml, pieces.xml or measurements.xml.
        If these is no tag you need, add new one and edit this info on the wiki page.
        It YOUR duty to make sure there is no duplicate tag in the xml files.
    2. Decide what are the extra info tags you would like to show. Add tags in the list, tableItem like:
        tableItem = ['sampleAuthor','history','pieceAuthor','measurementAuthor','otherFile']
    3. Rerun this script, then it will generate the html pages containing info table
        indicating where are the row data files and other info, e.g. an image or a histogram.



'''

#
# TODO: 1.check the file existing
#       2.check the id consistence
#       Done3.add general function could add xml texts by tags in the table
#       4.add duplicate Id checking function in another py script. import it
#       Done5.add __main__ and doc
#       6. modify doc


import xml.etree.ElementTree as ET

# decide the file format
pageFormat = ".html"
rowDataFormat = ".asc"
imageFormat = ".png"
tableItem = ['sampleAuthor','history','pieceAuthor','measurementAuthor','otherFile','sampleDimensions','pieceDimensions']

def MakeIndexHtmlPrototype(root):

    tree = ET.parse("sampleTitle.xml")
    sample = tree.find("sample")

    head =  ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    body =  ET.SubElement(root, "body")

    sampleTable = ET.SubElement(body, "table")
    sampleTableRowTitle = ET.SubElement(sampleTable,"tr")
    ET.SubElement(sampleTableRowTitle,"th").text = "sample chunk"
    ET.SubElement(sampleTableRowTitle,"th").text = "sample piece"
    for tag in tableItem:
        if sample.find(tag) != None:
            ET.SubElement(sampleTableRowTitle,"th").text = sample.find(tag).text
    title.text = "Dayabay acrylic transmittance measurement - sample chunks"
    sampleTable.set("border","1")

    return sampleTable
    pass

def MakePieceHtmlPrototype(root):

    tree = ET.parse("pieceTitle.xml")
    piece = tree.find("piece")

    head =  ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    body =  ET.SubElement(root, "body")

    sampleTable = ET.SubElement(body, "table")
    sampleTableRowTitle = ET.SubElement(sampleTable,"tr")
    ET.SubElement(sampleTableRowTitle,"th").text = "sample piece"
    ET.SubElement(sampleTableRowTitle,"th").text = "treatment and measurement"

    for tag in tableItem:
        if piece.find(tag) != None:
            ET.SubElement(sampleTableRowTitle,"th").text = piece.find(tag).text

    title.text = "Dayabay acrylic transmittance measurement - sample pieces of a chunk"
    sampleTable.set("border","1")

    return sampleTable
    pass

def MakeMeasurementHtmlPrototype(root):

    tree = ET.parse("measurementTitle.xml")
    measurement = tree.find("measurement")

    head =  ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    body =  ET.SubElement(root, "body")

    sampleTable = ET.SubElement(body, "table")
    sampleTableRowTitle = ET.SubElement(sampleTable,"tr")
    ET.SubElement(sampleTableRowTitle,"th").text = "treatment and measurement"
    ET.SubElement(sampleTableRowTitle,"th").text = "row data file"

    for tag in tableItem:
        if measurement.find(tag) != None:
            ET.SubElement(sampleTableRowTitle,"th").text = measurement.find(tag).text

    title.text = "Dayabay acrylic transmittance measurement - sample pieces measurement"
    sampleTable.set("border","1")

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
        # looping default column info
        for piece in sample.getiterator("piece"):
            linkFileName = sample.tag+sample.attrib.get("sampleId") + "_" + piece.tag + piece.attrib.get("pieceId") 
            href = ET.SubElement(ET.SubElement(colB,"p"),"a")
            href.set("href",linkFileName + pageFormat)
            if piece.text == None:
                href.text = linkFileName
            else: href.text = linkFileName + " " + piece.text
            MakePiecesPage(sample,piece,linkFileName + pageFormat)
        # looping option column info
        for tag in tableItem:
             if sample.find(tag) != None:
                 ET.SubElement(row,"td").text = sample.find(tag).text
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
            # looping default column info
            for measurement in piece.getiterator("measurement"):
                linkFileName = sample.tag + sample.attrib.get("sampleId") + "_"\
                                + piece.tag + piece.attrib.get("pieceId") + "_"\
                                + measurement.tag + measurement.attrib.get("measurementId")
                href = ET.SubElement(ET.SubElement(colB,"p"),"a")
                href.set("href",linkFileName + pageFormat)
                href.text = linkFileName
                if measurement.text == None:
                    href.text = linkFileName
                else: href.text = linkFileName + " " + measurement.text
                MakeMeasurementsPage(sample,piece,measurement,linkFileName + pageFormat)
            # looping option column info
            for tag in tableItem:
                if piece.find(tag) != None:
                    ET.SubElement(row,"td").text = piece.find(tag).text

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
            # looping default column info
            for rowData in measurement.getiterator("rowData"):
                linkFileName = sample.attrib.get("sampleId")\
                             + "-" + piece.attrib.get("pieceId")\
                             + "-" + measurement.attrib.get("measurementId")\
                             + "-" + rowData.attrib.get("rowDataId")
                href = ET.SubElement(ET.SubElement(colB,"p"),"a")
                href.set("href",linkFileName + rowDataFormat)
                href.text = linkFileName
                if rowData.text == None:
                    href.text = linkFileName
                else: href.text = linkFileName + " " + rowData.text
            # looping option column info
            for tag in tableItem:
                # tag "otherFile" is a special tag because it would provide a link
                if tag == "otherFile":
                    for file in measurement.getiterator(tag):
                        #linkFileName = file.attrib.get("fileName")
                        href = ET.SubElement(ET.SubElement(ET.SubElement(row,"td"),"p"),"a")
                        href.set("href",linkFileName)
                        href.text = file.attrib.get("fileName")
                        if file.text == None:
                            href.text = linkFileName
                        else: href.text = linkFileName + " " + file.text
                if measurement.find(tag) != None:
                    ET.SubElement(row,"td").text = measurement.find(tag).text

    outputTree = ET.ElementTree(root)
    outputTree.write(ifile)
    pass

if '__main__' == __name__:
    print __doc__
    MakeSamplesPage()
    # import checkId
    # CheckId()

