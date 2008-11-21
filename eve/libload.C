{
   gSystem->ListLibraries();
   gSystem->Load("libEve");
   gSystem->ListLibraries();
   gSystem->Load("$ENV_HOME/eve/InstallArea/$CMTCONFIG/lib/libSplitGLViewLib.so");
   gSystem->ListLibraries();
}


