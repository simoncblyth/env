import string
import xml.etree.ElementTree as ET





# generating the samples list
def MakeSamplePage():
    tree = ET.parse("samples.xml")
    ti = tree.getiterator("sample")
    plti = tree.getiterator("piecesLocation")
    root = ET.Element("html")
    head = ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    title.text = "Dayabay acrylic transmittance measurement"
    body = ET.SubElement(root, "body")
    for sample in ti:
    
        para = ET.SubElement(body, "p")
        sampleId = sample.attrib.get("id")
        print "sampleId is ", sampleId
        sampleFull = sample.tag+sample.attrib.get("id")
        para.text = sampleFull
    
        pie = plti[string.atoi(sampleId)]
        pieceLink = pie.attrib.get("href")
        piecehref = ET.SubElement(para,"a")
        piecehref.set("href",pieceLink)
        piecehref.text = pie.text

        outputTree = ET.ElementTree(root)
        outputTree.write("testpageSample.html")
    pass


# generating the pieces list
def MakePiecePage():
    pieceTree = ET.parse("pieces.xml")
    pieceTi = pieceTree.getiterator("testPiece")
    pieceRoot = ET.Element("html")
    pieceHead = ET.SubElement(pieceRoot, "head")
    pieceTitle = ET.SubElement(pieceHead, "title")
    pieceTitle.text = "Dayabay acrylic transmittance measurement"
    pieceBody = ET.SubElement(pieceRoot,"body")
    for piece in pieceTi:
    
        piecePara = ET.SubElement(pieceBody,"p")
        pieceName = piece.tag + " " + piece.attrib.get("pieceId") + " of sample " + piece.attrib.get("sampleId")
        print pieceName
    
        pieceParaMea = ET.SubElement(piecePara,"p")
        pieceParaMeahref = piece.getiterator("measurement")
        for mea in pieceParaMeahref:
            pieceParaMeaLink = ET.SubElement(pieceParaMea,"a")
            pieceParaMeaLink.set("href",mea.attrib.get("href"))
            pieceParaMeaLink.text = pieceName + " measurement " + mea.attrib.get("meaId")
    pieceOutPutTree = ET.ElementTree(pieceRoot)
    pieceOutPutTree.write("testpagePiece.html")
    pass


# generating the measurement list
def MakeMeasurementPage():
    pass







MakeSamplePage()
MakePiecePage()
MakeMeasurementPage()


