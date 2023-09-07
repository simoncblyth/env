bes3-env(){ echo -n ; }
bes3-vi(){ vi $BASH_SOURCE ; }

bes3-notes(){ cat << EON
BES3
=====

Dear Simon, 

This is the link to the article, and you have editing permission. Thank you very much for your help！

https://latex.ihep.ac.cn/8656799587zrxztssqzjps

Best regards, 
Sicheng




EON
}
bes3-cd(){ cd $(env-home)/presentation/bes3 ; }
bes3-wc(){ 
   bes3-cd
   wc -w *abstract.tex
}



