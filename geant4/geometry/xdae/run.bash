#!/bin/bash -l
nuwa-

out=daetest.gdml

rm $out
$(nuwa-g4-xdir)/xdae

head -20 $out



