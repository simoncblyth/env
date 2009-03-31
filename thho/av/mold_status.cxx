// A simple root script to analyze the status of IAV top lid steel mold


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


void initializeHypotenuse(double radiusPoints[]) {

    radiusPoints[0] = 75.0;
    radiusPoints[1] = 300.0;
    radiusPoints[2] = 600.0;
    radiusPoints[3] = 900.0;
    radiusPoints[4] = 1200.0;
    radiusPoints[5] = 1563.0;

}
/*
void drawDiff(double radiusPoints[],double line[], FitLinesParas* fitLinesParas) {

    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        for(int j=0;j<TOTALRINGNO;j++) {
            fitLinesParas[i].diff[j] = line[j] - (fitLinesParas[i].fitParas[0]+fitLinesParas[i].fitParas[1]*radiusPoints[j]);
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
    }

}
*/
TGraph* drawFitLine(double radiusPoints[], double line[], int lineNo, FitLinesParas* fitLinesParas) {

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
    //cout << "line address " << line << endl;
    TCanvas* cv = new TCanvas(c.data(),s.data(),200,10,700,500);
    TGraph* graph = new TGraph(TOTALRINGNO, radiusPoints, line);
    graph->Draw("a*");
    graph->SetMarkerColor(kBlue);
    graph->Fit("pol1");

    TF1* fitFunc = (TF1*) graph->GetFunction("pol1");
    fitFunc->GetParameters(fitLinesParas[lineNo].fitParas);
    // Debug.
    //cout << fitLinesParas << endl;
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

void mold_status() {

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

    FitLinesParas fitLinesParas[TOTALPOINTNOONRING];
    // Debug.
    //cout << "address of fitLinesParas " << fitLinesParas << endl;

    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        double noLine[TOTALRINGNO];
        // Debug.
        //cout << "address of noLine " << noLine << endl;
        fillLine(measurements, noLine, i);
        TGraph* gr = drawFitLine(radiusPoints, noLine, i, fitLinesParas);
        // Debug.
        //cout << "address of fitLinesParas in " << &fitLinesParas[0] << endl;
        // Debug.
        //cout << i << "th Line" << endl;
    }

    TCanvas* fitSlopCv = new TCanvas("fitSlop");
    TH1D* fitSlop = new TH1D("fitSlop","Fit Slops",30,2.7,3.0);
    for(int i=0;i<TOTALPOINTNOONRING;i++) {
        fitSlop->Fill(atan(fitLinesParas[i].fitParas[1])*(180.0/PI));
        // Debug.
        //cout << "fitLinesParas[i].fitParas[1] " << fitLinesParas[i].fitParas[1] << endl;
    }
    fitSlop->Draw();

    //drawDiff(radiusPoints, fitLinesParas);

}
