#!/bin/bash

read -p "How many the files are?: " nu
read -p "What's the energy? (MeV): " en
read -p "How many events each files?: " evt

model=origin
particle=gamma
case $en in
    "1")
        energy=1mev
        ;;
    "6")
        energy=6mev
        ;;
    "*")
        echo Wrong Input!
        ;;
esac

echo Model   :  $model
echo Particle:  $particle
echo Energy  :  $energy

echo Starting NuWa......
source $HOME/runNuWa.sh

cd $HOME/om_work/$model/$particle/$energy/


for (( i=1; i<=$nu; i=i+1 ))
    do
    echo Analyzing $particle\_$energy\_$i.root
    time nuwa.py -A None -n $evt env.thho.NuWa.AVTest.chkGamma $particle\_$energy\_$i.root > GammaBasicPlots_$i.log 2>&1
    mv GammaBasicPlots.root GammaBasicPlots_$i.root
    done

