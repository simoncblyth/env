chep-env(){ echo -n ; }
chep-e(){  chep-cd ; vi chep.txt ; }
chep-vi(){ vi $BASH_SOURCE ; }
chep-cd(){ cd $(env-home)/presentation/chep ; }

chep-wc(){ 
   chep-cd
   wc -w chep*_abstract.txt
}



