#!/bin/bash -l
nuwa-

dae=test.dae

rm $dae
$(nuwa-g4-xdir)/xdae
head -20 $dae
xmllint --noout --schema ../DAE/schema/collada_schema_1_4.xsd $dae



