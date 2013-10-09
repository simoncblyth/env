#!/bin/bash -l
nuwa-
nginx-

dae=$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae

rm $dae
$(nuwa-g4-xdir)/xdae
head -20 $dae
xmllint --noout --schema ../DAE/schema/collada_schema_1_4.xsd $dae

name=$(basename $dae)
cmd="cp $dae $(nginx-htdocs)/dae/$name"
url="http://belle7.nuu.edu.tw/dae/$name"
echo $cmd : makes the dae accessible at $url 
eval $cmd




