#define FIN_T "1-1-1-1.csv"
#define FIN_R "1-1-2-1.csv"
#define TOTALRAWDATANO 601

#define CAUCHY_A 1.47325
#define CAUCHY_B 6.15911e3

#define ATT_LOW 6.58737e2
#define ATT_UP 2.47079e3
#define ATT_CUT 3.51423e2
#define ATT_DELTA 6.50717
#define TOTALABSWL 270.0

#define DOUBLE_ZERO 1.0e-99
#define DOUBLE_INFINITE 1.0e99

#define PI 3.1415926
#define THICKNESS 10.18

#define SUCCESS 0
#define ERROR 1

class FresnelModel{

    public:
        FresnelModel();
        ~FresnelModel();
        void loadTDataFromFile(string finData);
        void loadRDataFromFile(string finData);
        FresnelModel(string finTra, string finRef);
        double evalCauchy(double wl, double cauchyA, double cauchyB);
        double evalAttenuation(double wl, double lowAtt, double upAtt, double cutAtt, double delta);
        double evalK(double wl, double att);
        double evalIntT(double att);
        double evalFSR(double ior, double wl);
        double evalFresnelT(double tra, double ref);
        double evalFresnelR(double tra, double ref);
        evalFresnel(void);
        compare(double *wlArr, double *dataArr, double *paraArr);
        compareT(void);
        compareR(void);

        int getSuccessSize(int *status);
        setSuccessSize(void);
        getSuccessData(double *dataWl, *data, *para, *dataRT, *dataFresnel);

    private:
        double dataWl_[TOTALRAWDATANO];
        double dataTra_[TOTALRAWDATANO];
        double dataRef_[TOTALRAWDATANO];
        double paraN_[TOTALRAWDATANO];
        double paraAtt_[TOTALRAWDATANO];
        double paraFresnelT_[TOTALRAWDATANO];
        double paraFresnelR_[TOTALRAWDATANO];

        int fresnelStatus_[TOTALRAWDATANO]; // indicate when the Fresnel model applys
        int successSize_;

};

FresnelModel::FresnelModel(void) {

}

FresnelModel::~FresnelModel(void) {

}

FresnelModel::loadTDataFromFile(string finData) {

    ifstream fin;
    fin.open(finData.data());
    for(int i=0;i<TOTALRAWDATANO;i++) {
        fin >> dataWl_[i] >> dataTra_[i];
    }

}

FresnelModel::loadRDataFromFile(string finData) {

    ifstream fin;
    fin.open(finData.data());
    for(int i=0;i<TOTALRAWDATANO;i++) {
        fin >> dataWl_[i] >> dataRef_[i];
    }

}

FresnelModel::FresnelModel(string finTra, string finRef) {

    loadTDataFromFile(finTra);
    loadRDataFromFile(finRef);

}



double FresnelModel::evalCauchy(double wl, double cauchyA, double cauchyB) {

    return cauchyA + cauchyB/(wl*wl);

}

double FresnelModel::evalAttenuation(double wl, double lowAtt, double upAtt, double cutAtt, double delta) {

    double attVal = (lowAtt - upAtt)/(1 + exp((wl - cutAtt)/delta)) + upAtt
                    - (TOTALABSWL/wl)*lowAtt;
    cout << attVal << endl;
    if((attVal<0.0) || (wl<TOTALABSWL)) attVal = 0.0; // absorptance domain, Fresnel model could not apply

    return attVal;

}

double FresnelModel::evalK(double wl, double att) {

    double kappa(0);
    if(att < DOUBLE_ZERO) { kappa = DOUBLE_INFINITE; } else { kappa = (wl*1.0e-6)/(4.0*PI*att); }
    return kappa;

}

double FresnelModel::evalIntT(double att) {

    double intT(0);
    if(att < DOUBLE_ZERO) { intT = 0.0; } else { intT = exp(-(1.0/att)*THICKNESS); }
    
    return intT;

}

double FresnelModel::evalFSR(double ior, double kappa) {

    return (((ior - 1.0)*(ior - 1.0) + kappa*kappa)/((ior + 1.0)*(ior + 1.0) + kappa*kappa));

}

double FresnelModel::evalFresnelT(double tra, double ref) {

    return ((1.0 - ref)*(1.0 - ref)*tra)/(1.0 - ref*ref*tra*tra);

}

double FresnelModel::evalFresnelR(double tra, double ref) {

    return ref*(1.0 + tra*evalFresnelT(tra, ref));

}

FresnelModel::evalFresnel(void) {

    for(int i=0;i<TOTALRAWDATANO;i++) {
        int wl = 800 - i;
        double ior = evalCauchy(wl, CAUCHY_A, CAUCHY_B);
        double att = evalAttenuation(wl, ATT_LOW, ATT_UP, ATT_CUT, ATT_DELTA);
        paraN_[i] = ior;
        paraAtt_[i] = att;
        double kappa = evalK(wl, att);
        double tra = evalIntT(att);
        double ref = evalFSR(ior, kappa);
        //cout << "wl" << wl << " ior " << ior << " att " << att << " kappa " << kappa << " evalIntT(ior, att) " << tra << " evalFSR(ior, att) " << ref << endl;
        paraFresnelT_[i] = 100.0*evalFresnelT(tra, ref); // times 100 for %
        paraFresnelR_[i] = 100.0*evalFresnelR(tra, ref); // times 100 for %

        //cout << "wl " << wl << " paraFresnelT_[i] " <<  paraFresnelT_[i] << " paraFresnelR_[i] " << paraFresnelR_[i] << endl << endl;
        if((kappa > (DOUBLE_INFINITE/3.0)) && (att < (DOUBLE_ZERO*3.0))) { fresnelStatus_[i] = ERROR; } else { fresnelStatus_[i] = SUCCESS;}
        //cout << wl << " " << fresnelStatus_[i] << endl;
    }

}

FresnelModel::compare(string type, double *wlArr, double *dataArr, double *paraArr, double lowBound, double upBound) {

    TCanvas *canv = new TCanvas(type.data(), type.data(), 200,10,700,450);

    const int successSize = successSize_;

    TH1F *hrR = canv->DrawFrame(0,lowBound,1000,upBound);
    TGraph *grData = new TGraph(successSize, wlArr, dataArr);
    grData->SetMarkerColor(kBlue);
    grData->Draw("LP");
    TGraph *grPara = new TGraph(successSize, wlArr, paraArr);
    grPara->SetMarkerColor(kRed);
    grPara->Draw("LP");

}

int FresnelModel::getSuccessSize(int *arr) {

    int counter(0);
    for(int i=0;i<TOTALRAWDATANO;i++) {
        if(arr[i] == 0) counter++;
    }
    return counter;

}

FresnelModel::setSuccessSize(void) {

    successSize_ = getSuccessSize(fresnelStatus_);

}

FresnelModel::getSuccessData(double *dataWl, double *data, double *para, double *dataRT, double *dataFresnel) {

    for(int i=0;i<TOTALRAWDATANO;i++) {
        if(fresnelStatus_[i] == 0) {
            dataWl[i] = dataWl_[i];
            data[i] = dataRT[i];
            para[i] = dataFresnel[i];
            //cout << "wl\tdata\tpara" << endl;
            //cout << 800 - i << " " << data[i] << " " << para[i] << endl;
        }
    }

}

FresnelModel::compareT(void) {

    const int successSize = successSize_;
    double dataWl[successSize], data[successSize], para[successSize];
    getSuccessData(dataWl, data, para, dataTra_, paraFresnelT_);
    compare("transmission", dataWl, data, para, 0, 100);

}

FresnelModel::compareR(void) {

    const int successSize = successSize_;
    double dataWl[successSize], data[successSize], para[successSize];
    getSuccessData(dataWl, data, para, dataRef_, paraFresnelR_);
    compare("reflection", dataWl, data, para, 0, 10);

}

void compareFresnelModel(void) {

    FresnelModel fresnelModel(FIN_T, FIN_R);
    //FresnelModel fresnelModel("1-1-1-1.csv","1-1-2-1.csv");
    fresnelModel.evalFresnel();
    fresnelModel.setSuccessSize();
    fresnelModel.compareT();
    fresnelModel.compareR();


}
