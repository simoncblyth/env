# TODO: 1.check the file existing
#       2.check the id consistence
#       3.add general function could add xml texts by tags in the table


import xml.etree.ElementTree as ET

fileFormat=".html"

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
        rowA = ET.SubElement(sampleTable,"tr")
        ET.SubElement(rowA,"td").text = sample.tag+sample.attrib.get("sampleId")
        colB = ET.SubElement(rowA,"td")
        for piece in sample.getiterator("piece"):
            linkFileName = sample.tag+sample.attrib.get("sampleId") + "_" + piece.tag + piece.attrib.get("pieceId") 
            href = ET.SubElement(ET.SubElement(colB,"p"),"a")
            href.set("href",linkFileName + fileFormat)
            href.text = linkFileName
            MakePiecesPage(sample,piece,linkFileName + fileFormat)

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
        rowA = ET.SubElement(sampleTable,"tr")
        # should check the sampleId in sample.xml and piece.xml consistent,not yet complete
        if (isample.attrib.get("sampleId") == piece.attrib.get("sampleId")) and \
           (ipiece.attrib.get("pieceId") == piece.attrib.get("pieceId")):
            sample = isample
            ET.SubElement(rowA,"td").text = sample.tag + sample.attrib.get("sampleId") + "_" + piece.tag + piece.attrib.get("pieceId")
            colB = ET.SubElement(rowA,"td")
            for measurement in piece.getiterator("measurement"):
                linkFileName = piece.tag + piece.attrib.get("pieceId") + "_" + measurement.tag + measurement.attrib.get("measurementId")
                MakeMeasurementsPage(sample,piece,measurement,linkFileName + fileFormat)
                href = ET.SubElement(ET.SubElement(colB,"p"),"a")
                href.set("href",linkFileName + fileFormat)
                href.text = linkFileName

    outputTree = ET.ElementTree(root)
    outputTree.write(ifile)
    pass



# generating the measurement list
def MakeMeasurementsPage(isample,ipiece,imeasurment,ifile):
    root = ET.Element("html")
    sampleTable = MakeMeasurementHtmlPrototype(root)

    tree = ET.parse("measurement.xml")
    ti = tree.getiterator("measurement")

    for measurement in ti:
        rowA = ET.SubElement(sampleTable,"tr")
        print measurment
        # should check the sampleId in sample.xml and piece.xml consistent,not yet complete
        if (isample.attrib.get("sampleId") == measurment.attrib.get("sampleId")) and \
           (ipiece.attrib.get("pieceId") == measurment.attrib.get("pieceId")) and \
           (imeasurment.attrib.get("measurmentId") == measurement.attrib.get("measurementId")):
            sample = isample
            piece = ipiece
            ET.SubElement(rowA,"td").text = sample.tag + sample.attrib.get("sampleId") + "_" \
                                            + piece.tag + piece.attrib.get("pieceId") + "_" \
                                            + measurment.tag + measurment.attrib.get("measurmentId")
            colB = ET.SubElement(rowA,"td")
            for rowData in measurment.getiterator("rowData"):
                linkFileName = sample.attrib.get("sampleId") + piece.attrib.get("pieceId") + measurement.attrib.get("measurementId")
                href = ET.SubElement(ET.SubElement(colB,"p"),"a")
                href.set("href",linkFileName + fileFormat)
                href.text = linkFileName

    outputTree = ET.ElementTree(root)
    outputTree.write(ifile)
    pass

# general function to access xml data by tag and put it in the table
def GetXmlTextByTag():
    pass

MakeSamplesPage()


