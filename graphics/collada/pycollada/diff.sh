#!/bin/sh

xmllint --format orig.dae > origf.dae
xmllint --format copy.dae > copyf.dae
diff origf.dae copyf.dae > dif.txt


