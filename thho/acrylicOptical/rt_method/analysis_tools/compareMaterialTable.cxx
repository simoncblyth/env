/*************************************************\

compareMaterialModel.cxx

a modification code of compareFresnelModel.cxx
so the code look weird XD


Compare the original materail table parameters with
customized material table




\*************************************************/

#define FIN_T "originATT.dat"
#define FIN_R "originIOR.dat"
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
        evalFresnel(void);
        compare(double *wlArr, double *dataArr, double *paraArr);
        compareT(void);
        compareR(void);

        int getSuccessSize(int *status);
        setSuccessSize(void);
        getSuccessData(double *dataWl, *data, *para, *dataRT, *dataFresnel);

    private:
        double dataWl_[TOTALRAWDATANO];
        double dataWlIOR_[9];
        double dataWlATT_[7];
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
    for(int i=0;i<7;i++) {
        fin >> dataWlATT_[i] >> dataTra_[i];
    }

    for(int i=0;i<7;i++) {
        dataWlATT_[i] = 1200.0/dataWlATT_[i];
        //cout << "ATT " << dataWlATT_[i] << " " << dataTra_[i] << endl;
    }


}

FresnelModel::loadRDataFromFile(string finData) {

    ifstream fin;
    fin.open(finData.data());
    for(int i=0;i<9;i++) {
        fin >> dataWlIOR_[i] >> dataRef_[i];
    }

    for(int i=0;i<9;i++) {
        dataWlIOR_[i] = 1200.0/dataWlIOR_[i];
        //cout << "IOR " << dataWlIOR_[i] << " " << dataRef_[i] << endl;
    }


}

FresnelModel::FresnelModel(string finTra, string finRef) {

    loadTDataFromFile(finTra);
    loadRDataFromFile(finRef);
    for(int i=0;i<TOTALRAWDATANO;i++) dataWl_[i] = 800.0 - i;


}



double FresnelModel::evalCauchy(double wl, double cauchyA, double cauchyB) {

    return cauchyA + cauchyB/(wl*wl);

}

double FresnelModel::evalAttenuation(double wl, double lowAtt, double upAtt, double cutAtt, double delta) {

    double attVal = (lowAtt - upAtt)/(1 + exp((wl - cutAtt)/delta)) + upAtt
                    - (TOTALABSWL/wl)*lowAtt;
    //cout << attVal << endl;
    if((attVal<0.0) || (wl<TOTALABSWL)) attVal = 0.0; // absorptance domain, Fresnel model could not apply

    return attVal;

}
FresnelModel::evalFresnel(void) {

    for(int i=0;i<TOTALRAWDATANO;i++) {
        int wl = 800 - i;
        double ior = evalCauchy(wl, CAUCHY_A, CAUCHY_B);
        double att = evalAttenuation(wl, ATT_LOW, ATT_UP, ATT_CUT, ATT_DELTA);
        paraN_[i] = ior;
        paraAtt_[i] = att;
        paraFresnelT_[i] = att;
        paraFresnelR_[i] = ior;

        if(att>0) {fresnelStatus_[i] = SUCCESS;} else { fresnelStatus_[i] = ERROR; }
    }

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

FresnelModel::getSuccessData(double *dataWl, double *para, double *dataFresnel) {

    for(int i=0;i<TOTALRAWDATANO;i++) {
        if(fresnelStatus_[i] == 0) {
            dataWl[i] = dataWl_[i];
            para[i] = dataFresnel[i];
            //cout << "SUCCESS DATA " << dataWl[i] << " " << para[i] << endl;
        }
    }

}

FresnelModel::compareT(void) {

    const int successSize = successSize_;
    double dataWl[successSize], para[successSize];
    getSuccessData(dataWl, para, paraFresnelT_);

    TCanvas *canv = new TCanvas("Attenuation_Length", "Attenuation_Length", 200,10,700,450);

    TH1F *hrR = canv->DrawFrame(0,0,1000,5500);
    TGraph *grData = new TGraph(7, dataWlATT_, dataTra_);
    grData->SetMarkerColor(kBlue);
    grData->Draw("L*");
    TGraph *grPara = new TGraph(successSize, dataWl, para);
    //cout << "SUCCESS DATA " << dataWl[77] << " " << para[77] << endl;
    grPara->SetMarkerColor(kRed);
    grPara->Draw("LP");

}

FresnelModel::compareR(void) {

    const int successSize = successSize_;
    double dataWl[successSize], para[successSize];
    getSuccessData(dataWl, para, paraFresnelR_);

    TCanvas *canv = new TCanvas("Index_of_Refraction", "Index_of_Refraction", 200,10,700,450);

    TH1F *hrR = canv->DrawFrame(0,1.4,1000,1.7);
    TGraph *grData = new TGraph(9, dataWlIOR_, dataRef_);
    grData->SetMarkerColor(kBlue);
    grData->Draw("L*");
    TGraph *grPara = new TGraph(successSize, dataWl, para);
    grPara->SetMarkerColor(kRed);
    grPara->Draw("LP");
}

void compareMaterialTable(void) {

    FresnelModel fresnelModel(FIN_T, FIN_R);
    //FresnelModel fresnelModel("1-1-1-1.csv","1-1-2-1.csv");
    fresnelModel.evalFresnel();
    fresnelModel.setSuccessSize();
    fresnelModel.compareT();
    fresnelModel.compareR();


}
