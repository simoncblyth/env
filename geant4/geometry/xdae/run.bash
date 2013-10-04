#!/bin/bash -l
nuwa-

out=test.dae

rm $out
$(nuwa-g4-xdir)/xdae

head -20 $out



