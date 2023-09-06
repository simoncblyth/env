cepc-env(){ echo -n ; }
cepc-vi(){ vi $BASH_SOURCE ; }

cepc-notes(){ cat << EON
CEPC 
=====

2023 CEPC Workshop in Nanjing
-----------------------------

The 2023 international workshop on the high energy Circular Electron-Positron
Collider (CEPC) will take place at Nanjing, Oct 23-27, 2023. Detailed
information can be found at https://indico.ihep.ac.cn/event/19316 The deadline
of registration is Oct 1, 2023.

Please follow and register for the 2023 CEPC Workshop in Nanjing. Please note
the deadline for submission of abstracts. Looking forward to your
participation.


https://indico.ihep.ac.cn/event/19316/

The abstract submission deadline is Sept 16, 2023.

https://indico.ihep.ac.cn/event/19316/abstracts/

Your abstract 'Opticks : GPU Optical Photon Simulation via NVIDIA OptiX' has
been successfully submitted. It is registered with the number #62. You will be
notified by email with the submission details. 



EON
}
cepc-cd(){ cd $(env-home)/presentation/cepc ; }
cepc-wc(){ 
   cepc-cd
   wc -w *abstract.tex
}



