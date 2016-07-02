#!/bin/bash
#===============================================================
#
# This shell script emulates the behaviour of the 'cmt buid dependencies' operation
# but using cpp -M instead
#
# Nathaniel, Dyb, Nov 14/06: don't use cpp -M, use g++ -E -M
#
#===============================================================

function compute_dependencies ()
{
  file=$1

  # Generate the expected format:
  #  one single line
  #  prefix is a make macro
  #  ends with a dependency to the stamp file

  a=`eval g++ -E -M ${allflags} ${file} | \
    sed -e 's#[.]o:#_'"${suffix}"'_dependencies = #' -e 's#[\\]$##'`

  if test ! `echo ${a} | wc -w` = "0"; then
    echo -n ${a}
  else
    echo -n "${file_name}_${suffix}_dependencies = ${file}"
  fi

  echo " \$(bin)${constituent}_deps/${file_name}_${suffix}.stamp"

  # create or update the stamp file

  touch ${bin}/${constituent}_deps/${file_name}_${suffix}.stamp
}

#--------------------------------------------
macro_value ()
{
  name=$1
  shift

  grep "^${name}=" ${tempmacros} | sed -e "s#^${name}=##"
}

#-----------------------------------------------------
# Pre-compute all configuration parameters from CMT queries
#-----------------------------------------------------
function prepare_context ()
{
  /bin/rm -f ${tempmacros}
  cmt -quiet build tag_makefile > ${tempmacros}
  cmt -quiet filter ${tempmacros} ${tempmacros}A; mv ${tempmacros}A ${tempmacros}

  # /bin/rm -f ${tempconstituents}
  # cmt -quiet show constituents > ${tempconstituents}
  # cmt -quiet filter ${tempconstituents} ${tempconstituents}A; mv ${tempconstituents}A ${tempconstituents}
}

#------------------------------------------------------------------------------------------
# Main
#
#  Expected arguments:
#    1    : <constituent name>
#    2    : -all_sources
#    3... : <source file list>
#
#------------------------------------------------------------------------------------------

constituent=$1
shift

all_sources=$1
shift

files=$*

#---------------
# Prepare temporary file management
#
tempprefix=/tmp/CMT$$
if test ! "${TMP}" = ""; then
  tempprefix=${TMP}/CMT$$
fi

tempmacros=${tempprefix}/macros$$
tempconstituents=${tempprefix}/constituents$$

trap "if test -d ${tempprefix} ; then chmod -R +w ${tempprefix}; fi; /bin/rm -rf ${tempprefix}" 0 1 2 15

if test -d ${tempprefix} ; then chmod -R +w ${tempprefix}; fi
/bin/rm -rf ${tempprefix}
mkdir -p ${tempprefix}
#---------------

#---------------
# prepare the context from CMT
#
prepare_context

incl=`macro_value includes`

cflags=`macro_value cflags`
const_cflags=`macro_value ${constituent}_cflags`
const_pp_cflags=`macro_value ${constituent}_pp_cflags`

cppflags=`macro_value cppflags`
const_cppflags=`macro_value ${constituent}_cppflags`
const_pp_cppflags=`macro_value ${constituent}_pp_cppflags`

bin=`macro_value bin`
#--------------

#--------------
# Prepare the directory for the stamp files
#
mkdir -p ${bin}/${constituent}_deps
#--------------

#--------------
# Prepare the dependency file
#
output=${bin}${constituent}_dependencies.make
#--------------

#--------------
# Loop over source files (if any)
#
for f in `echo ${files}`; do
  suffix=`echo ${f} | sed -e 's#.*[.]##'`
  file_name=`basename ${f} .${suffix}`

  # First remove the old dependency line from the output

  if test -f ${output}; then
    grep -v "${file_name}_${suffix}_dependencies" ${output} >t$$; mv t$$ ${output}
  fi

  echo "computing dependencies for ${file_name}.${suffix}"

  case ${suffix} in
    c ) allflags="${incl} ${cflags} ${const_cflags} ${const_cpp_cflags}";;
    C|cc|cxx|cpp ) allflags="${incl} ${cppflags} ${const_cppflags} ${const_cpp_cppflags}";;
  esac

  echo "cpp -M ${allflags} ${f}"

  compute_dependencies ${f} >>${output}
done
#--------------



