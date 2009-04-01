//
// Demo C++ codes for analyzing mold of IAV top lids status.
// Compile this codes with
//
// g++ $(root-config --cflags --libs) mold_status.cpp -o mold_status
//
// This codes haven't been fully completed.
// 
//
// Author: Taihsiang
// Date: 2009,Apr,01
//


#include <iostream>
#include <sstream>
#include <fstream>
#include "TH1D.h"
#include "TGraph.h"
#include "TCanvas.h"

using namespace std;

#define TOTALRINGNO 6
#define TOTALPOINTNOONRING 12
#define BASESIDESTEP 300
#define PI 3.1415926

//void loadFromFile()

//double getHypotenuseOneEnd(void) { return 0;}

typedef struct
{
    double fitParas[2];

}FitLinesParas;

void initializeHypotenuse(double radiusPoints[]) {

    radiusPoints[0] = 75.0;
    radiusPoints[1] = 300.0;
    radiusPoints[2] = 600.0;
    radiusPoints[3] = 900.0;
    radiusPoints[4] = 1200.0;
    radiusPoints[5] = 1563.0;

}

TGraph* drawFitLine(double radiusPoints[], double line[], int lineNo, FitLinesParas fitLinesParas[]) {

    stringstream ss;
    ss << lineNo;
    string s = "Line ";
    s += ss.str();
    s += " Slop";
    string c = "cv";
    c += ss.str();
    // Debug.
    //cout << s << endl;
    //cout << c << endl;
    cout << "line address " << line << endl;
    TCanvas* cv = new TCanvas(c.data(),s.data(),200,10,700,500);;
    TGraph* graph = new TGraph(TOTALRINGNO, radiusPoints, line);
    graph->Draw("a*");
    graph->SetMarkerColor(kBlue);
    graph->Fit("pol1");

    //TF1* fitFunc = (TF1*) graph->GetFunction("pol1");
    //fitFunc->GetParameters(fitLinesParas[lineNo].fitParas);
    // Debug.
    cout << fitLinesParas << endl;
    //cout << fitLinesParas[lineNo].fitParas[1] << endl;
    //cout << "The angle is(degree) " << atan(fitLinesParas[lineNo].fitParas[1])*(180.0/PI) << endl;

    return graph;

}

void fillLine(double measurements[TOTALRINGNO][TOTALPOINTNOONRING], double line[], int lineNo) {

    for(int i=0;i<TOTALRINGNO;i++) { 
        line[i] = measurements[i][lineNo];
        // Debug.
        //cout << line[i] << " ";
    }
    // Debug.
    //cout << endl;

}

void mold_status(void) {

    ifstream fin;
    fin.open("mold.dat");

    double measurements[TOTALRINGNO][TOTALPOINTNOONRING];
    double radiusPoints[TOTALRINGNO];
    initializeHypotenuse(radiusPoints);

    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        // Debug.
        //cout << i << " ";
        for(int j=0;j<TOTALRINGNO;j++) {
            fin >> measurements[j][i];
            // Debug.
            //cout << i << " " << j << " ";
            //cout << measurements[j][i] << " ";
        }
        // Debug.
        //cout << endl;
    }


    FitLinesParas fitLinesParas[12];
    // Debug.
    cout << "address of fitLinesParas " << fitLinesParas << endl;

    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        double noLine[TOTALRINGNO];
        // Debug.
        //cout << "address of noLine " << noLine << endl;
        fillLine(measurements, noLine, i);
        TGraph* gr = drawFitLine(radiusPoints, noLine, i, fitLinesParas);
        // Debug.
        //cout << i << "th Line" << endl;
    }
/*
    TCanvas* fitSlopCv = new TCanvas("fitSlop");
    TH1D* fitSlop = new TH1D("fitSlop","Fit Slops",1000,2.9,3.0);
    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        fitSlop->Fill(fitLinesParas[i].fitParas[1]);
        // Debug.
        cout << "fitLinesParas[i].fitParas[1] " << fitLinesParas[i].fitParas[1] << endl;
    }
    fitSlop->Draw();
*/

}

int main(void) {


    mold_status();
    return 0;


}
