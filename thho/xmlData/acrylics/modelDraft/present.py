import elementtree.ElementTree as ET
tree = ET.parse("samples.xml")
ti =tree.getiterator()
print ti[0]
