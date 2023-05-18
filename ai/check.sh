#!/bin/bash -l 

name=check

[ -z "$OPENAI_API_KEY"  ] && echo $BASH_SOURCE error need OPENAI_API_KEY && exit 1 

echo $BASH_SOURCE using OPENAI_API_KEY $OPENAI_API_KEY

${IPYTHON:-ipython} --pdb -i $name.py 
