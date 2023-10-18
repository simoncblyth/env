cepc-env(){ echo -n ; }
cepc-vi(){ vi $BASH_SOURCE ; }

cepc-notes(){ cat << EON
CEPC 
=====

The 2023 international workshop on the high energy Circular Electron-Positron
Collider (CEPC) will take place at Nanjing, Oct 23-27, 2023. The opening
session will be held in the Science and Technology Building of Nanjing
University at its Gulou Campus. All other sessions will be at the Grand Hotel
Nanjing.

The abstract submission deadline is Sept 16, 2023. The registration deadline is
Oct 1, 2023. 


Presentation : 30 mins : What to include ?
---------------------------------------------

Its in "Offline and Software" session : so focus on software aspects. 




Registration
--------------

* https://indico.ihep.ac.cn/event/19316/registrations/1581/

Arrival: Oct 22, 2023 (Sun)
Departure: Oct 28, 2023 (Sat)
Accommodation: Nanjing Grand Hotel (Single Room) ￥450/night 

Registration fee:
- 2000 CNY/person for staff and postdocs
- 1000 CNY/person for students
- No registration fee for online participants



CEPC Talk : Fri Oct 29 : 09:00-09:30 
---------------------------------------

* https://indico.ihep.ac.cn/event/19316/timetable/
* https://indico.ihep.ac.cn/event/19316/sessions/12209/#20231027


Opticks : GPU Optical Photon Simulation via NVIDIA OptiX
CEPC Room 4, GrandHotelNanjing
09:00 - 09:30


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



