#!/bin/bash -l

cmdline="$*"

oops-usage(){ cat << EOU
oops : Opticks Operations
===========================

oops.sh is intended to replace ggv.sh using 
simplifications possible following the
move to the superbuild approach.

EOU
}

oops-binary-name-default(){ echo GGeoView ; }
oops-binary-name()
{
   case $1 in 
            --cmp) echo computeTest ;; 
     --boundaries) echo BoundariesNPYTest ;;
           --cfg4) echo cfg4Test ;;
       --cproplib) echo CPropLibTest ;;
      --cdetector) echo CDetectorTest ;;
           --recs) echo RecordsNPYTest ;;
         --tracer) echo OTracerTest ;;
         --lookup) echo LookupTest ;;
            --bnd) echo GBndLibTest ;;
       --itemlist) echo GItemListTest ;;
        --gsource) echo GSourceTest ;;
        --gsrclib) echo GSourceLibTest ;;
       --resource) echo OpticksResourceTest ;;
        --opticks) echo OpticksTest ;;
          --pybnd) echo GBndLibTest.py ;;
            --mat) echo GMaterialLibTest ;;
             --mm) echo GMergedMeshTest ;;
        --testbox) echo GTestBoxTest ;;
         --geolib) echo GGeoLibTest ;;
        --geotest) echo GGeoTestTest ;;
         --gmaker) echo GMakerTest ;;
            --pmt) echo GPmtTest ;;
           --attr) echo GAttrSeqTest ;;
           --surf) echo GSurfaceLibTest ;;
         --tscint) echo GScintillatorLibTest ;;
         --oscint) echo OScintillatorLibTest ;;
          --flags) echo GFlagsTest ;;
        --gbuffer) echo GBufferTest ;;
           --meta) echo GBoundaryLibMetadataTest ;;
         --sensor) echo GSensorListTest ;;
           --ggeo) echo GGeoTest ;;
         --assimp) echo AssimpRapTest ;;
       --openmesh) echo OpenMeshRapTest ;;
      --torchstep) echo TorchStepNPYTest ;;  
           --hits) echo HitsNPYTest ;;  
   esac 
}

oops-geometry-name()
{
   case $1 in 
       --dyb)  echo DYB ;; 
       --idyb) echo IDYB ;; 
       --jdyb) echo JDYB ;; 
       --kdyb) echo KDYB ;; 
       --ldyb) echo LDYB ;; 
       --mdyb) echo MDYB ;; 
       --juno) echo JUNO ;; 
       --jpmt) echo JPMT ;; 
       --jtst) echo JTST ;; 
       --dpib) echo DPIB ;; 
       --dpmt) echo DPMT ;; 
   esac
}

oops-geometry-setup()
{
    local geo=${OPTICKS_GEO:-DYB}
    oops-geometry-unset 
    case $geo in 
       DYB|IDYB|JDYB|KDYB|LDYB|MDYB) oops-geometry-setup-dyb  $geo  ;;
                     JUNO|JPMT|JTST) oops-geometry-setup-juno $geo  ;;
                          DPIB|DPMT) oops-geometry-setup-dpib $geo  ;;
    esac
}

oops-geometry-query-dyb()
{
    case $1 in 
        DYB)  echo "range:3153:12221"  ;;
       IDYB)  echo "range:3158:3160" ;;  # 2 volumes : pvIAV and pvGDS
       JDYB)  echo "range:3158:3159" ;;  # 1 volume : pvIAV
       KDYB)  echo "range:3159:3160" ;;  # 1 volume : pvGDS
       LDYB)  echo "range:3156:3157" ;;  # 1 volume : pvOAV
       MDYB)  echo "range:3201:3202,range:3153:3154"  ;;  # 2 volumes : first pmt-hemi-cathode and ADE  
    esac
    # range:3154:3155  SST  Stainless Steel/IWSWater not a good choice for an envelope, just get BULK_ABSORB without going anywhere
}

oops-geometry-setup-dyb()
{
    local geo=${1:-DYB}
    export OPTICKS_GEOKEY=DAE_NAME_DYB
    export OPTICKS_QUERY=$(oops-geometry-query-dyb $geo) 
    export OPTICKS_CTRL="volnames"
    export OPTICKS_MESHFIX="iav,oav"
    export OPTICKS_MESHFIX_CFG="100,100,10,-0.999"   # face barycenter xyz alignment and dot face normal cuts for faces to be removed 
}
oops-geometry-setup-juno()
{
   local geo=${1:-JPMT}
   if [ "$geo" == "JUNO" ]; then 
       export OPTICKS_GEOKEY=DAE_NAME_JUNO
       export OPTICKS_QUERY="range:1:50000"
       export OPTICKS_CTRL=""
   elif [ "$geo" == "JPMT" ]; then
       export OPTICKS_GEOKEY=DAE_NAME_JPMT
       export OPTICKS_QUERY="range:1:289734"  # 289733+1 all test3.dae volumes
       export OPTICKS_CTRL=""
   elif [ "$geo" == "JTST" ]; then
       export OPTICKS_GEOKEY=DAE_NAME_JTST
       export OPTICKS_QUERY="range:1:50000" 
       export OPTICKS_CTRL=""
   fi
}
oops-geometry-setup-dpib()
{
   local geo=${1:-DPIB}
   if [ "$geo" == "DPIB" ]; then
       export OPTICKS_GEOKEY=DAE_NAME_DPIB
       export OPTICKS_QUERY="" 
       export OPTICKS_CTRL=""
    elif [ "$geo" == "DPMT" ]; then
       export OPTICKS_GEOKEY=DAE_NAME_DPIB
       export OPTICKS_QUERY="range:1:6"   # exclude the box at first slot   
       export OPTICKS_CTRL=""
   fi 
}
oops-geometry-unset()
{
    unset OPTICKS_GEOKEY
    unset OPTICKS_QUERY 
    unset OPTICKS_CTRL
    unset OPTICKS_MESHFIX
    unset OPTICKS_MESHFIX_CFG
}






oops-cmdline-dump()
{
    >&2 echo $0 $FUNCNAME
    local arg
    for arg in $cmdline 
    do
       if [ "${arg/=}" == "${arg}" ]; then  
           >&2 printf "%s\n" $arg
       else
           oops-dump _ $arg
       fi
    done
}
oops-dump(){
  local IFS="$1" ; shift  
  local elements
  read -ra elements <<< "$*" 
  local elem 
  for elem in "${elements[@]}"; do
      >&2 printf "   %s\n" $elem
  done 
}


oops-cmdline-specials()
{
   unset OPTICKS_DBG 
   unset OPTICKS_LOAD
   unset OPTIX_API_CAPTURE

   if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
       export OPTICKS_DBG=1
   fi
   if [ "${cmdline/--load}" != "${cmdline}" ]; then
       export OPTICKS_LOAD=1
   fi
   if [ "${cmdline/--oac}" != "${cmdline}" ]; then
       export OPTIX_API_CAPTURE=1
   fi
}

oops-cmdline-binary-match()
{
    local msg="=== $FUNCNAME : finds 1st argument with associated binary :"
    local arg
    local bin
    unset OPTICKS_CMD
    for arg in $cmdline 
    do
       bin=$(oops-binary-name $arg)
       #echo arg $arg bin $bin geo $geo 
       if [ "$bin" != "" ]; then 
           export OPTICKS_CMD=$arg
           return 
       fi
    done
}


oops-binary-setup()
{
    local msg="=== $FUNCNAME :"
    local cfm=$OPTICKS_CMD
    local bin=$(oops-binary-name $cfm) 
    local def=$(oops-binary-name-default)

    if [ "$bin" == "" ]; then
       bin=$def
    fi 
    #echo $msg cfm $cfm bin $bin def $def

    unset OPTICKS_BINARY 
    unset OPTICKS_ARGS

    if [ "$bin" != "" ]; then
       export OPTICKS_BINARY=$(opticks-bindir)/$bin
       export OPTICKS_ARGS=${cmdline/$cfm}
    fi 
}


oops-cmdline-geometry-match()
{
    local msg="=== $FUNCNAME : finds 1st argument with associated geometry :"
    local arg
    local geo
    unset OPTICKS_GEO
    for arg in $cmdline 
    do
       geo=$(oops-geometry-name $arg)
       #echo arg $arg geo $geo 
       if [ "$geo" != "" ]; then 
           export OPTICKS_GEO=$geo
           return 
       fi
    done
}







oops-cmdline-parse()
{
    #oops-cmdline-dump
    oops-cmdline-specials

    oops-cmdline-binary-match
    oops-cmdline-geometry-match

    oops-binary-setup
    oops-geometry-setup
}


oops-export()
{
   export-
   export-export
}

oops-runline()
{
   local runline
   if [ "${OPTICKS_BINARY: -3}" == ".py" ]; then
      runline="python ${OPTICKS_BINARY} ${OPTICKS_ARGS} "
   elif [ "${OPTICKS_DBG}" == "1" ]; then 
      case $(uname) in
          Darwin) runline="lldb ${OPTICKS_BINARY} -- ${OPTICKS_ARGS} " ;;
               *) runline="gdb  ${OPTICKS_BINARY} -- ${OPTICKS_ARGS} " ;;
      esac
   else
      runline="${OPTICKS_BINARY} -- ${OPTICKS_ARGS}" 
   fi
   echo $runline
}


opticks-
oops-cmdline-parse
env | grep OPTICKS

runline=$(oops-runline)
echo $runline

oops-export
eval $runline





