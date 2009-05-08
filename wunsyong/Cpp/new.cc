#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <iostream.h>
#include <fstream.h>
 // #include <cstdlib.h>
#define NPARA           2             // number of parameters
#define MaxEvent        1000000
#define length(i)       event[i][0]      // name of parameters
#define trans(i)       event[i][1]
#define CMM(b)         (strncasecmp(tmp,b,strlen(b))==0&&strlen(b)==strlen(tmp))



int nevent=0;  // number of event
double event[MaxEvent][NPARA]; // event array
char sourcefile[100]="./data";
char filename[80]="haha";
int read_data();
double equation(double x);
void loadoptions();
double distance = 10;
double eq_result[1000];


int main()
{
	loadoptions();

	cout << "The distance is  " << distance << endl;
	cout << endl;

        int read_result;
        read_result = read_data();                        //read from data file

//	for(int i=0;i<nevent;i++) { if(trans(i)<0) trans(i) = -trans(i); }


        if (read_result==0){
                cout <<"succeed, ";
                cout <<nevent<<" events loaded."<<endl;
                           }
        else {
                cout <<"failed!"<<endl;
                exit(0);
             }

	for(int i=0;i<nevent;i++)  eq_result[i] = equation(trans(i));
	
	for(int i=0;i<nevent;i++)  cout<< "trans (" << i <<") : " << trans(i) << "  pass EQ: " << eq_result[i] << endl; 



	cout << "Output file :  " << filename << endl;

	ofstream fout(filename);

	for(int i=0;i<nevent;i++)	fout << eq_result[i]  << endl;
	
	fout.close();
}


//==============================================//
// read data from file                          //
//==============================================//
int read_data()
{

        cout<<"Reading from "<<sourcefile<<" ...";
        FILE *datafile,*outfile;


        datafile=fopen(sourcefile,"r");
//        outfile=fopen("selected.2d","w");
        if (datafile == NULL) return 1;
        cout<<"I am still alive"<<endl;

        while(nevent<MaxEvent)
        {
                int err;
                for (int i=0;i<NPARA;i++){
                        char tmp[64];
                        err= fscanf(datafile,"%s",tmp);
                        if (err!=1) break;
                        event[nevent][i]=atof(tmp);
                }
                if(err!=1) break;

                nevent++;
                if (nevent%50==0) cout <<".";

        } //end of "  while(nevent<MaxEvent)  "
        cout <<endl;
        cout <<"Total events : " <<nevent<<endl;

//    for(int i=0; i<nevent; i++)
 //   {
//        data->column("mp1",mp1(i));
//        data->column("mp2",mp2(i));

//        data->dumpData();
//    }

    return 0;

}






double equation(double x)
{
	return -log(x/100)/distance;
}



void loadoptions()
{
                        FILE *ctrlfile=fopen("option.txt","r");
                        int err,flag=0;

                        do {
                                char tmp[128],tmp2[128];
                                double v;
                                if (flag==0){
                                        err = fscanf(ctrlfile,"%s",tmp);
                                        if (err!=1) break;
                                        if (CMM("Options"))
                                        {
                                                cout <<"Loading Program options...";
                                                flag=1;
                                        }
                                } else
                                {
                                        err = fscanf(ctrlfile,"%s",tmp);
                                        if (CMM("EndOptions"))
                                        {
                                                cout <<"Options loaded."<<endl;
                                                flag=0;
                                                err=1;
                                                break;

                                        }
                                        if (err!=1) break;
                                        err = fscanf(ctrlfile,"%s",tmp2);
                                        if (err!=1) break;

                                        if(CMM("/*"))
                                        {
                                                do {
                                                        err = fscanf(ctrlfile,"%s",tmp);
                                                        if (err!=1) cout <<"comment braket error!"<<endl;
                                                } while(!CMM("*/"));

                                        }
                                        else
                                        {
					v=atof(tmp2);
                                            if(CMM("distance")) distance=v;
                                            if(CMM("sourcefile")) strcpy(sourcefile,tmp2);
                                            if(CMM("output_file")) strcpy(filename,tmp2);

                                        }
                                }

                        } while(err==1);
}

