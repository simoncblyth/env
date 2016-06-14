function cmak-src { $script:MyInvocation.MyCommand.Path }
function cmak-sdir { Split-Path -Parent -Path $script:MyInvocation.MyCommand.Path }
function cmak-scd { cd $(cmak-sdir) }
function cmak-vi{   vim $(cmak-src) }
function cmak-usage { Write-Host @"

cmak powershell module
================================

::

   src       $(cmak-src) 
   sdir      $(cmak-sdir) 


CAUTION: powershell/cmake 

* very sensitive to precise incantation ie quoting, escaping, variable replacement in arguments
* doesnt work with drive letters on paths

Dev cycle for a function::

    imp cmak ; cmak-test

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




"@}



function cmak-dir{ "${env:LOCAL_BASE}\env\tools\cmak"  }

function cmak-cd{  
   $dir = $(cmak-dir)
   md -force $dir > $null
   cd $dir
}

function cmak-txt 
{ 
   param([string]$name="XercesC")

   echo @"
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(tt)

set(CMAKE_MODULE_PATH `$`{CMAKE_MODULE_PATH`}
                      `$ENV`{ENV_HOME`}/cmake/Modules
)

find_package($name REQUIRED)
"@
}



function cmak-txt-write
{
   param([string]$name="XercesC")
   cmak-txt $name  | Out-File -Encoding ascii  CMakeLists.txt

   # huh byte order marker problem with simple redirection to file 
}


function cmak-build-prep
{
   $bnam = "build"
   if([io.directory]::exists($bnam))
   {
      rd -R -force $bnam > $null
   }
   mkdir $bnam > $null
}


function cmak-test-glew
{
   Import-Module glew -DisableNameChecking 

   cmak-cd
   cmak-txt-write GLEW

   cmak-build-prep
   cd build > $null
 
   cmake `
           "$(cmak-dir)"   

}


function cmak-test-xercesc
{
   Import-Module xercesc -DisableNameChecking 

   cmak-cd
   cmak-txt-write XercesC

   cmak-build-prep
   cd build > $null
 
   #       --trace `
   cmake `
           "-DXercesC_LIBRARY=$(xercesc-lib)" `
           "-DXercesC_INCLUDE_DIR=$(xercesc-include)" `
           "$(cmak-dir)"   

}



