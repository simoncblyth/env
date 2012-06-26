

dyw-rootcint-kludge-usage(){ cat << EOU

EOU
}

dyw-rootcint-kludge(){

  ## the bug that caused this kludge is claimed to be fixed in root version > 5.14
  if [ "$CMTBIN" == "Linux-x86_64" ]; then
      cmd0="echo SCB rootcint kludge"
	  cmd1="perl -pi -e 's/Logging::\*fptr/\*fptr/;' \$@ "
	  frag=$DYW/External/ROOT/cmt/fragments/rootcint
	  grep kludge $frag || printf "\t%s\n\t%s\n" "$cmd0" "$cmd1" >> $frag
  else
	  echo dyw-rootcint-kludge is not needed on $CMTBIN
  fi
}

