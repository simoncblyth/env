#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>

#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAEChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEPhotons.hh"

#include <cassert>

/*




 not exist?  /usr/local/env/tmp/20140514-180740.npy 
 not exist?  /usr/local/env/tmp/20140514-180800.npy 
 reg /usr/local/env/tmp/3.npy 
 reg /usr/local/env/tmp/mock001.npy 
 not exist?  /usr/local/env/tmp/mock002.npy 


*/


bool canwrite( const char* path )
{
    bool ret = false ;      

    struct stat st; 
    lstat(path, &st);

    if( S_ISDIR(st.st_mode) )
    {
        //printf(" dir %s \n", dest );
    }
    else if (S_ISREG(st.st_mode))
    {
        //printf(" reg %s \n", dest );
    }
    else 
    {
        ret = true ; 
    }
    return ret ; 
}




int convert( const char* srcpath )
{
    // NB must use copy ctor in order to write to other format 

    G4DAEPhotons* photons = G4DAEPhotons::LoadPhotons(srcpath);
    //photons->Print();

    std::string destpath ; 
    if(G4DAEPhotons::HasExt(srcpath, ".root"))
    {
        G4DAEChromaPhotonList* gcpl = dynamic_cast<G4DAEChromaPhotonList*>(photons) ;
        assert(gcpl);
        destpath = G4DAEPhotons::SwapExt(srcpath, ".root", ".npy");
        if(canwrite(destpath.c_str()))
        {
            printf("writing .npy to %s \n" , destpath.c_str() );
            G4DAEPhotonList* gpl = new G4DAEPhotonList(gcpl); 
            gpl->SavePath( destpath.c_str());
        }
        else
        {
            printf("skip writing .npy to %s as exists already \n" , destpath.c_str() );
        }
    }
    else if(G4DAEPhotons::HasExt(srcpath, ".npy"))
    {
        G4DAEPhotonList* gpl = dynamic_cast<G4DAEPhotonList*>(photons) ;
        assert(gpl);
        destpath = G4DAEPhotons::SwapExt(srcpath, ".npy", ".root");

        if(canwrite(destpath.c_str()))
        {
            printf("writing .root to %s \n" , destpath.c_str() );
            G4DAEChromaPhotonList* gcpl = new G4DAEChromaPhotonList(gpl); 
            gcpl->SavePath( destpath.c_str());
        }
        else
        {
            printf("skip writing .root to %s as exists already \n" , destpath.c_str() );
        }
    }
    delete photons ; 

    return 0 ; 
}



int recursive_listdir( const char* base, const char* fext )
{
    DIR* dir ; 
    struct dirent* dent;
    struct stat st; 
    char path[1024];
    char xath[1024];

    if (!(dir = opendir(base)))
        return -1; 

    while ((dent = readdir(dir)) != NULL) 
    {   
        int len = snprintf(path, sizeof(path)-1, "%s/%s", base, dent->d_name);
        path[len] = 0;

        lstat(path, &st);

        if(S_ISDIR(st.st_mode))
        {   
            if(strcmp(dent->d_name,".") == 0 || strcmp(dent->d_name,"..") == 0 ) continue ;
            else
            {   
                //printf("\t D %s \n", path );
                recursive_listdir( path, fext );  
            }   
        }   
        else
        {   
            int flen = strlen(dent->d_name);
            if( flen < 4 ){
                printf("\t F SKIP short filename  %s \n", path );
                continue ; 
            }   


            const char* flast = dent->d_name + flen - strlen(fext)  ; 
            if( strcmp( flast, fext ) == 0 ){
                  // 
                  //printf("\t found %s  %s \n", fext, path );
                  convert( path );
            }   
            else 
            {   
                  //printf("\t OTHER [%s] %s \n", flast, path );
            }   

        }   
    }   
    closedir(dir); 
    return 0;
}



int main(int argc, char** argv)
{
    const char* base = "/usr/local/env/tmp" ;
    const char* fext = ".root" ; 
    recursive_listdir( base, fext  );
}
