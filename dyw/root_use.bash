

root-use-conf(){

   ## ftp://root.cern.ch/root/doc/chapter2.pdf
   local rootrc=$HOME/.rootrc
   local rootlogon=$HOME/.rootlogon.C
   local names="rootrc rootlogon"
   
   for name in $names
   do
        eval path=\$$name
        if [ -f "$path" ]; then
            echo === not proceeding, as $path exists ...  first delete $path 
        else
            echo === writing $path
            root-use-$name > $path
            cat $path
        fi
   done
   
}

root-use-rootrc(){

cat << EOC
#
# do not edit $HOME/.rootrc  , see source:/trunk/dyw/root_use.bash
# $ROOTSYS/etc/system.rootrc for details
#
Unix.*.Root.MacroPath:      .:\$(ROOTSYS)/macros:$HOME/$ENV_BASE/root
EOC

}

root-use-rootlogon(){

cat << EOL
{

    TString CMTBIN = gSystem->Getenv("CMTBIN") ;
    CMTBIN = CMTBIN.Contains("Darwin") ? "Darwin" : CMTBIN ;
    cout << "$HOME/.rootlogon.C loading (created by root-use-rootlogon, invoked by root-use-conf, see env:trunk/dyw/root_use.bash ) " << endl ;
    gSystem->Load( "$DYW/InstallArea/" + CMTBIN + "/lib/libMCEvent.so" );
    
    gStyle->SetOptStat(1111111);
     //see http://root.cern.ch/root/htmldoc/THistPainter.html#THistPainter:Paint

     TString HOME(gSystem->Getenv("HOME"));
    // 
    //gROOT->ProcessLine(".L view_geom.C");  
    //gROOT->ProcessLine(".L dump_geom.C");  
    
}
EOL

}