# === func-gen- : virtualization/vgpu fgp virtualization/vgpu.bash fgn vgpu fgh virtualization
vgpu-src(){      echo virtualization/vgpu.bash ; }
vgpu-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vgpu-src)} ; }
vgpu-vi(){       vi $(vgpu-source) ; }
vgpu-env(){      elocal- ; }
vgpu-usage(){ cat << EOU

NVIDIA VGPU
==============



VMware and NVIDIA announced at VMworld 2018 today a technology preview of vSphere vMotion for NVIDIA GPUs with VMware vSphere 6.7 update 1
-------------------------------------------------------------------------------------------------------------------------------------------

This is possible because NVIDIA’s virtualization software — NVIDIA Quadro
Virtual Data Center Workstation (Quadro vDWS) and NVIDIA GRID — run on the same
Tesla Volta and Pascal GPUs as deep learning, inferencing, training and HPC
workloads. When graphics resources are needed for VDI the next day, IT admins
can simply repurpose the NVIDIA GPUs to virtual GPUs to support VDI again.



* https://blogs.nvidia.com/blog/2018/08/27/gpu-live-migration-vmotion-virtualization/

Additionally, vSphere 6.7 Update 1 will increase support for intelligent
workloads by introducing vMotion and snapshot capabilities for NVIDIA Quadro
vDWS powered VMs. This will enable admins to migrate vGPU-powered VMs to other
compatible hosts while performing maintenance operations to completely
eliminate any disruption to end-users and their applications.


virtual desktop infrastructure (VDI) market



vGPU : Quadro vDWS (Quadro Virtual Data Center Workstation) : required for CUDA
----------------------------------------------------------------------------------

* https://www.nvidia.com/en-us/design-visualization/technologies/virtual-gpu/



GRID vPC, Quadro vDWS and GRID vApps are available on a per Concurrent User
(CCU) model. A CCU license is required for every user who is accessing or using
the software at any given time, whether or not an active connection to the
virtualized desktop or session is maintained.  

NVIDIA vGPU editions can be
purchased as either perpetual licenses with annual Support Updates and
Maintenance Subscription (SUMS), or as an annual subscription. The first year
of SUMS is required with purchase of a perpetual license and can then be
purchased as a yearly subscription. For annual licenses SUMS is bundled into
the annual license cost.  vGPU Software Pricing is listed in the tables below,
find the full SKU list here. Pricing is suggested pricing only, contact your
authorized NVIDIA partner for final pricing.


NVIDIA EDUCATION PRICING PROGRAM 

The NVIDIA Education Pricing Program supports
the use of visual computing in teaching and research institutions. The program
makes it easy to procure and administer and helps reduce the total cost of
NVIDIA Solutions, software licensing and services for qualified educational
institutions. The program includes NVIDIA solutions, software and services. For
more information on eligibility, please review the NVIDIA Education Pricing
Program documentation.
   
Annual Subscription Pricing
 
Quadro Virtual Data Center Workstation
 
$50 per CCU subscription

Perpetual Licensing + SUMS Pricing

Quadro Virtual Data Center Workstation
 
$99 per CCU perpetual license
$25 SUMS











* http://www.zillians.com/how-it-works-2/




EOU
}
vgpu-dir(){ echo $(local-base)/env/virtualization/virtualization-vgpu ; }
vgpu-cd(){  cd $(vgpu-dir); }
vgpu-mate(){ mate $(vgpu-dir) ; }
vgpu-get(){
   local dir=$(dirname $(vgpu-dir)) &&  mkdir -p $dir && cd $dir

}
