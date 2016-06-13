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









