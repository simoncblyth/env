#include <iostream>

#include <stdio.h>
#include <stdlib.h>

extern char **environ;

int main (int argc, char * const argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n" << std::endl ;
    
    char * pPath;
    pPath = getenv ("PATH");
    if (pPath!=NULL)
        printf ("The current path is: %s",pPath);
    
    std::cout << "dump the environment ... " << std::endl ;
    char  **envp;
    for (envp = environ; envp && *envp; envp++)
        printf("%s\n", *envp);
    

    return 0;
    
}

