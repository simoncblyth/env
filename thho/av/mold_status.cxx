//
// A simple root script to analyze the status of IAV top lid steel mold
//
// Author: Taihsiang
// Date: 2009,Apr,01
//

#define TOTALRINGNO 6
#define TOTALPOINTNOONRING 12
#define BASESIDESTEP 300
#define PI 3.1415926

//void loadFromFile()

//double getHypotenuseOneEnd(void) { return 0;}

typedef struct
{
    double fitParas[2];
    double diff[TOTALRINGNO];

}FitLinesParas;

typedef struct
{
    double points[TOTALRINGNO];
}LineData;

void initializeHypotenuse(double radiusPoints[]) {

    radiusPoints[0] = 75.0;
    radiusPoints[1] = 300.0;
    radiusPoints[2] = 600.0;
    radiusPoints[3] = 900.0;
    radiusPoints[4] = 1200.0;
    radiusPoints[5] = 1563.0;

}


void drawDiff(double radiusPoints[], LineData* lineData, FitLinesParas* fitLinesParas) {


    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        for(int j=0;j<TOTALRINGNO;j++) {
            fitLinesParas[i].diff[j] = lineData[i].points[j] - (fitLinesParas[i].fitParas[0]+fitLinesParas[i].fitParas[1]*radiusPoints[j]);
            // Debug.
            //cout << fitLinesParas[i].diff[j] << endl;
        }
        stringstream ss;
        ss << i;
        string s = "Line ";
        s += ss.str();
        s += " Diff";
        string c = "cvDiff";
        c += ss.str();
        TCanvas* cv = new TCanvas(c.data(),s.data(),200,10,700,500);
        TGraph* graph = new TGraph(TOTALRINGNO, radiusPoints, &fitLinesParas[i].diff[0]);
        graph->Draw("ACP");
        graph->SetMarkerColor(kBlue);
    }

}

void drawFitLine(double radiusPoints[], LineData* lineData, FitLinesParas* fitLinesParas) {

    for(int lineNo=0;lineNo<TOTALPOINTNOONRING;lineNo++) { 
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
        TCanvas* cv = new TCanvas(c.data(),s.data(),200,10,700,500);
        TGraph* graph = new TGraph(TOTALRINGNO, radiusPoints, &lineData[lineNo].points[0]);
        graph->Draw("a*");
        graph->SetMarkerColor(kBlue);
        graph->Fit("pol1");
    
        TF1* fitFunc = (TF1*) graph->GetFunction("pol1");
        fitFunc->GetParameters(fitLinesParas[lineNo].fitParas);
        // Debug.
        //cout << fitLinesParas << endl;
        //cout << fitLinesParas[lineNo].fitParas[1] << endl;
        //cout << "The angle is(degree) " << atan(fitLinesParas[lineNo].fitParas[1])*(180.0/PI) << endl;
    }


}

void mold_status() {

    // read in measurement data, storing in them in lines
    ifstream fin;
    fin.open("mold.dat");
    double radiusPoints[TOTALRINGNO];
    initializeHypotenuse(radiusPoints);
    FitLinesParas fitLinesParas[TOTALPOINTNOONRING];
    LineData lineData[TOTALPOINTNOONRING];
    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        for(int j=0;j<TOTALRINGNO;j++) {
            fin >> lineData[i].points[j];
        }
    }

    drawFitLine(radiusPoints, lineData, fitLinesParas);

    TCanvas* fitSlopCv = new TCanvas("fitSlop");
    TH1D* fitSlop = new TH1D("fitSlop","Fit Slops",30,2.7,3.0);
    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        fitSlop->Fill(atan(fitLinesParas[i].fitParas[1])*(180.0/PI));
        // Debug.
        //cout << "fitLinesParas[i].fitParas[1] " << fitLinesParas[i].fitParas[1] << endl;
        cout << "Line No " << i << ", Angle(degree) " << atan(fitLinesParas[i].fitParas[1])*(180.0/PI) << endl;
    }
    fitSlop->Draw();


    drawDiff(radiusPoints, lineData, fitLinesParas);

    TCanvas* moldContourCv = new TCanvas("moldContourCv");
    TH2D* moldContour = new TH2D("moldContour","Mold Contour",TOTALRINGNO,0,TOTALRINGNO,TOTALPOINTNOONRING,0,TOTALPOINTNOONRING);
    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        for(int j=0;j<TOTALRINGNO;j++) {
            moldContour->SetBinContent(j+1,i+1,fitLinesParas[i].diff[j]);
            //moldContour->SetBinContent(j,i,lineData[i].points[j]);
            //moldContour->SetBinContent(1,1,-1);
        }
    }
    /*
    gStyle->SetPalette(1);
    double contours[6];
    contours[0] = -1.0;
    contours[1] = -0.5;
    contours[2] = -0.1;
    contours[3] =  0.1;
    contours[4] =  0.5;
    contours[5] =  1.0;
    moldContour->SetContour(6, contours);
    */
    //moldContour->Draw("CONT Z LIST");
    moldContour->Draw("lego2");

}
