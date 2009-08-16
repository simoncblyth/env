#!/bin/bash

workdir=$HOME/home/om_work
codedir=$HOME/env/thho/acrylicOptical/analysis
nudefault=5
evtdefault=1000

om-chkGamma(){
    local nu=$nudefault
    local evt=$evtdefault
    if [ "$2" == "1mev" ]; then
        nu=1
        evt=5000
    fi
    for (( i=1; i<=$nu; i=i+1 ))
        do
        echo Checking $1\_$2\_$i.root
        echo executing......
        echo "time nuwa.py -A None -n $evt env.thho.NuWa.AVTest.chkGamma $1\_$2\_$i.root"
        time nuwa.py -A None -n $evt env.thho.NuWa.AVTest.chkGamma $1\_$2\_$i.root > GammaBasicPlots_$i.log 2>&1
        mv -f GammaBasicPlots.root GammaBasicPlots_$i.root
        done
}

om-chkIBD(){
    local nu=$nudefault
    for (( i=1; i<=$nu; i=i+1 ))
        do
        echo Checking $1\_$i.root
        echo executing......
        echo "time nuwa.py -A None -n $evtdefault env.thho.NuWa.AVTest.chkIBD $1\_$i.root"
        time nuwa.py -A None -n $evtdefault env.thho.NuWa.AVTest.chkIBD $1\_$i.root > IbdBasicPlots_$i.log 2>&1
        mv -f IbdBasicPlots.root IbdBasicPlots_$i.root
        done
}

om-anaGamma(){
    
    #local workin=$workdir/$1/$2/$3
    local analysis=analysisOM_$3_$2.cxx
    cd $workin
    if [ -L $workin/$analysis ]; then
        echo remove $workin/$analysis
        rm -f $workin/$analysis
    fi
    echo linking $codedir/$analysis
    ln -s $codedir/$analysis ./

    cd $workin
    echo Starting analyzing data......
    root -l $codedir/$analysis
    
}

om-anaIBD(){

    #local workin=$workdir/$1/$2
    local analysis=analysisOM_$2.cxx
    cd $workin
    if [ -L $workin/$analysis ]; then
        echo remove $workin/$analysis
        rm -f $workin/$analysis
    fi
    echo linking $codedir/$analysis
    ln -s $codedir/$analysis ./

    cd $workin
    echo Starting analyzing data......
    root -l $codedir/$analysis
}

om-runGamma() {

    local workin=$workdir/$1/$2/$3
    cd $workin
    om-chkGamma $2 $3
    om-anaGamma $1 $2 $3

}

om-runIBD() {

    local workin=$workdir/$1/$2
    cd $workin
    om-chkIBD $2
    om-anaIBD $1 $2

}

om-run() {
    echo "**********************************************"
    echo "**********************************************"
    echo Processing model $1
    om-runGamma $1 gamma 1mev > $HOME/om_log/$1\_gamma_1mev.log 2>&1
    om-runGamma $1 gamma 6mev > $HOME/om_log/$1\_gamma_6mev.log 2>&1
    om-runIBD $1 ibd > $HOME/om_log/$1\_ibd.log 2>&1
    echo Processing model $1 done!
    echo "**********************************************"
    echo "**********************************************"
}

cd $HOME
echo "Starting NuWa......"
source runNuWa.sh

om-run origin
om-run model_A
om-run model_B
om-run model_C
