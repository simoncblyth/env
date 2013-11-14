#!/bin/bash -l

dae-
[ ! -f orig.xml ] && cp `dae-pth` orig.xml 
[ ! -f origf.xml ] && xmllint --format orig.xml > origf.xml
xmllint --format 0.xml > 0f.xml
diff origf.xml 0f.xml > dif.txt


