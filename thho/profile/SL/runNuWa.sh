#!/bin/bash

echo "Hanging up dybgaudi......"
cd $DDR
source setup.sh
echo "Hanging up dybgaudi......done!"

echo "Hanging up DybRelease......"
cd $DDR/dybgaudi/DybRelease/cmt
source setup.sh
echo "Hanging up DybRelease......done!"

echo $NUWA_HOME is used!

cd $HOME
