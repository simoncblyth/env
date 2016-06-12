Import-Module xercesc -DisableNameChecking 


function cmakewin-src{      "$env:userprofile\env\tools\cmakewin.ps1" }
function cmakewin-vi{       vi $(cmakewin-src)  }
function cmakewin-usage{ echo @" 

Learning how to use CMake on windows
========================================


Minimal CMakeLists.txt::

    cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
    project(tt)
    find_package(XercesC REQUIRED)


    PS C:\usr\local\env\tt> cmake .\CMakeLists.txt
    -- Building for: Visual Studio 14 2015
    -- The C compiler identification is MSVC 19.0.23918.0
    -- The CXX compiler identification is MSVC 19.0.23918.0
    -- Check for working C compiler using: Visual Studio 14 2015
    -- Check for working C compiler using: Visual Studio 14 2015 -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working CXX compiler using: Visual Studio 14 2015
    -- Check for working CXX compiler using: Visual Studio 14 2015 -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    CMake Error at C:/ProgramData/chocolatey/lib/cmake.portable/tools/cmake-3.5.2-win32-x86/share/cmake-3.5/Modules/FindPackageHandleStandardArgs.cmake:148 (message):
      Failed to find XercesC (missing: XercesC_LIBRARY XercesC_INCLUDE_DIR
      XercesC_VERSION)
    Call Stack (most recent call first):
      C:/ProgramData/chocolatey/lib/cmake.portable/tools/cmake-3.5.2-win32-x86/share/cmake-3.5/Modules/FindPackageHandleStandardArgs.cmake:388 (_FPHSA_FAILURE_MESSAGE)
      C:/ProgramData/chocolatey/lib/cmake.portable/tools/cmake-3.5.2-win32-x86/share/cmake-3.5/Modules/FindXercesC.cmake:101 (FIND_PACKAGE_HANDLE_STANDARD_ARGS)
      CMakeLists.txt:5 (find_package)

    -- Configuring incomplete, errors occurred!
    See also "C:/usr/local/env/tt/CMakeFiles/CMakeOutput.log".

"@

}
function cmakewin-dir{ "${env:LOCAL_BASE}\env\tools\cmakewin"  }
function cmakewin-cd{  
   $dir = $(cmakewin-dir)
   md -force $dir
   cd $dir
}

function cmakewin-minimal{ echo @"
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(tt)
find_package(XercesC REQUIRED)
"@
}


function cmakewin-cmd { @"
cmake $(cmakewin-dir) -DXercesC_LIBRARY=$(xercesc-lib) -DXercesC_INCLUDE_DIR=$(xercesc-include)
"@
}



function cmakewin-test
{
cmakewin-cd

cmakewin-minimal | Out-File -Encoding "UTF8"  CMakeLists.txt
# huh byte order problem have to do in editor

rd -R -force build
mkdir build
cd build

echo $(cmakewin-cmd)

Invoke-Command -ScriptBlock { $(cmakewin-cmd) }


}









