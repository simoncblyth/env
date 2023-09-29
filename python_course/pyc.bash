pyc-vi(){ vi $BASH_SOURCE ; }
pyc-env(){ echo -n ; }
pyc-usage(){ cat << EOU
Python Course Ideas
============================

~/env/python_course/


1. Opticks input_photon.py as demo of NumPy arrays   

2. grab for Mandelbrot

3. grab U4Mesh for some shapes


Message to students
---------------------

To benefit fully from my presentation on C++ testing 
and python tools you need to setup a python environment
on your laptops. The easiest way to do that is to use
the anaconda distribution. Follow instructions from  
https://www.anaconda.com/ for your platform.

As this involves large downloads that could be slow 
you need to do this preparation work ahead of time.
The result is you will have a python development environment 
that you can use to learn and explore python packages
for data science and machine learning. 

How to download and install anaconda ? 
Visit the official anaconda website: https://www.anaconda.com/download
Read the instructions for you OS and download the appropriate installer.
There are also several mirrors, such as:

* Tsinghua University: https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D 
* Aliyun: https://mirrors.aliyun.com/anaconda/archive/

If there is any network issue to download the installers, 
you can also find the installers at IHEP cluster, in 
the directory: /besfs5/users/lint/anaconda-installers

* Windows, 64 bit: Anaconda3-2023.07-2-Windows-x86_64.exe
* Mac, 64 bit (Intel): Anaconda3-2023.07-2-MacOSX-x86_64.pkg
* Mac, 64 bit (M1): Anaconda3-2023.07-2-MacOSX-arm64.pkg
* Linux: Anaconda3-2023.07-2-Linux-x86_64.sh



Plan
-----

1. go thru the introductory slides including the Mandelbrot exercise

  * students will need to do things such as running  

2. go thru selected items from PythonDataScienceHandbook 

  * https://jakevdp.github.io/PythonDataScienceHandbook/
  * https://code.ihep.ac.cn/blyth/pythondatasciencehandbook
  * (ihep link incase access to github is blocked) 

  * i will be copy/pasting extracts from the handbook 
    into a ipython session and running the commands : 
    as well as describing what is happening and 
    how best to use NumPy and IPython functionality 

  * students will benefit most if they are able to do what I am 
    doing as I do it 

    * they will need to copy/paste snippets of code from the 
      PythonDataScienceHandbook html pages into ipython sessions
      running locally on their laptops (NOT remote server running)

The objective of the lesson is for students to get comfortable
with the ipython environment and how to play around and find
things by themselves. 

The point is NOT to learn specific commands : it is to 
learn how to find commands themselves.

IPython is the ideal environment to get familiar with python packages

The objective it is to become comfortable at following resources 
such the PythonDataScienceHandbook and running the code found there.

 
 


Backup
-------

* https://simoncblyth.bitbucket.io/env/presentation/opticks_gpu_optical_photon_simulation_oct2018_ihep.html


EOU
}

pyc-cd(){ cd ~/env/python_course ; }


