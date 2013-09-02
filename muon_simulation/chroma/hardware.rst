Finding Hardware
=================

Possibilities

* http://grid.ntu.edu.tw/html/contacts.html
* http://graphics.im.ntu.edu.tw/
* :google:`National Center of High Performance Computer NTU`
* :google:`ww.cc.ntu.edu.tw GPU`
* http://translate.google.com.tw/translate?hl=en&sl=zh-TW&u=http://www.cc.ntu.edu.tw/chinese/epaper/0012/20100320_1205.htm&prev=/search%3Fq%3Dww.cc.ntu.edu.tw%2BGPU%26safe%3Doff%26client%3Dsafari%26rls%3Den



lspci checking GPU type
------------------------

Needs to be recent Nvidia

::

    [blyth@belle7 LCG]$ /sbin/lspci | grep VGA
    01:00.0 VGA compatible controller: ATI Technologies Inc RV610 video device [Radeon HD 2400 PRO]



CUDA compute capability
-------------------------

http://en.wikipedia.org/wiki/CUDA

::

    1.3  [GT200, GT200b] 
     
       GeForce GTX 260, GTX 275, GTX 280, GTX 285, GTX 295, 
       Tesla C/M1060, S1070, 
       Quadro CX, FX 3/4/5800

    2.0 [GF100, GF110]   

       GeForce (GF100) GTX 465, GTX 470, GTX 480, 
       Tesla C2050, C2070, S/M2050/70, 
       Quadro Plex 7000, 
       Quadro 4000, 5000, 6000, 
       GeForce (GF110) GTX 560 TI 448, GTX570, GTX580, GTX590


