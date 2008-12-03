import elementtree.ElementTree as ET
tree = ET.parse("samples.xml")
ti =tree.getiterator("sample")
root = ET.Element("html")
head = ET.SubElement(root, "head")
title = ET.SubElement(head, "title")
title.text = "Dayabay acrylic transmittance measurement"
body = ET.SubElement(root, "body")
for sample in ti:
    para = ET.SubElement(body, "p")
    sampleName = sample.tag
    sampleId = sample.attrib.get("id")
    sampleFull = sampleName+sampleId
    para.text = sampleFull

outputTree = ET.ElementTree(root)
outputTree.write("testpage.html")


