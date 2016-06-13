Import-Module xercesc -DisableNameChecking 

function cmakewin-src{      "$env:userprofile\env\tools\cmakewin.ps1" }
function cmakewin-vi{       vi $(cmakewin-src)  }
function cmakewin-usage{ echo @" 

Learning how to use CMake on windows
========================================


CAUTION: powershell/cmake 

* very sensitive to precise incantation ie quoting, escaping, variable replacement in arguments
* doesnt work with drive letters on paths

Dev cycle for a function::

    . $(cmakewin-src) ; cmakewin-test


Observe that cmake seems to handle backslashes in paths so long 
as there is no drive letter.  Some however have needed to translate...

* http://stackoverflow.com/questions/4948121/problems-with-cmake-vars


Example of non-trival PS build system

* https://github.com/TheOneRing/appVeyorHelp/blob/master/appveyorHelp.psm1

How not to structure an installer, but some sytactical techniques

* https://github.com/phrasz/A5_Installer/blob/master/Installer.ps1


Syntax:

* http://www.neolisk.com/techblog/powershell-specialcharactersandtokens

* http://ss64.com/ps/syntax-operators.html



"@

}
function cmakewin-dir{ "${env:LOCAL_BASE}\env\tools\cmakewin"  }
function cmakewin-dir-not-working{ Split-Path -Parent -Path $MyInvocation.MyCommand.Definition }
# only works from modules in PS v2

function cmakewin-cd{  
   $dir = $(cmakewin-dir)
   md -force $dir > $null
   cd $dir
}

# the below function does not populate the scope of the caller 
# with the new function defns ?
# although running the command does so
# function cmakewin-{ . $(cmakewin-src) }
#
# ni -p function: -n cmakewin- -va ". $env:userprofile\env\tools\cmakewin.ps1"
#

function cmakewin-minimal{ echo @"
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(tt)
find_package(XercesC REQUIRED)
"@
}


function cmakewin-test
{
   cmakewin-cd

   cmakewin-minimal | Out-File -Encoding "UTF8"  CMakeLists.txt

   # huh byte order marker problem with simple redirection to file 

   rd -R -force build > $null
   mkdir build > $null
   cd build > $null
 
   #       --trace `
   cmake `
           "-DXercesC_LIBRARY=$(xercesc-lib)" `
           "-DXercesC_INCLUDE_DIR=$(xercesc-include)" `
           "$(cmakewin-dir)"   

}









