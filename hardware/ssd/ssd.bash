# === func-gen- : hardware/ssd/ssd fgp hardware/ssd/ssd.bash fgn ssd fgh hardware/ssd
ssd-src(){      echo hardware/ssd/ssd.bash ; }
ssd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ssd-src)} ; }
ssd-vi(){       vi $(ssd-source) ; }
ssd-env(){      elocal- ; }
ssd-usage(){ cat << EOU


SSD
=====


NVME : looks like a big performance step (Sep 2016)
-------------------------------------------------------

Comparisons
~~~~~~~~~~~~~

* http://www.tomshardware.com/reviews/best-ssds,3891.html
* http://www.anandtech.com/show/9799/best-ssds


Intel 750 series
~~~~~~~~~~~~~~~~~~~

* http://www.anandtech.com/show/9090/intel-ssd-750-pcie-ssd-review-nvme-for-the-client
* http://www.tomshardware.com/reviews/intel-ssd-750-800gb-nvme,4507.html

Intel 600p series
~~~~~~~~~~~~~~~~~~~~

* http://www.anandtech.com/show/10598/intel-launches-3d-nand-ssds-for-client-and-enterprise
* http://www.pcper.com/reviews/Storage/Intel-SSD-600p-Series-256GB-Full-Review-Low-Cost-M2-NVMe/Conclusion-Pricing-and-Fina

Samsung
~~~~~~~~~~

* http://www.tomshardware.com/reviews/samsung-sm961-ssd,4608.html
* http://www.tomshardware.com/reviews/samsung-sm961-ssd-256gb-512gb,4621-4.html
* http://www.anandtech.com/show/10437/samsung-sm961-price-and-availability-outlook



EOU
}
ssd-dir(){ echo $(local-base)/env/hardware/ssd/hardware/ssd-ssd ; }
ssd-cd(){  cd $(ssd-dir); }
ssd-mate(){ mate $(ssd-dir) ; }
ssd-get(){
   local dir=$(dirname $(ssd-dir)) &&  mkdir -p $dir && cd $dir

}
