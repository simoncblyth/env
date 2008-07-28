#!/bin/bash
#Discription:
#	an easy script to change many file names
for f in *; do
	mv $f TDC$f
done
