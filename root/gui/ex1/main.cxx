//  
//  example from http://root.cern.ch/root/hepvis98/newgui.html
//
// Compile example using:
// g++ `root-config --cflags --glibs` -o main main.cxx MyMain.cxx
//
// File: main.cxx

#include <TApplication.h> 
#include <TGClient.h>
#include "MyMain.h"
 int main(int argc, char **argv) 
{ 
    TApplication theApp("App", &argc, argv);
    MyMainFrame mainWin(gClient->GetRoot(), 200, 220);
    theApp.Run();
    return 0;
}


