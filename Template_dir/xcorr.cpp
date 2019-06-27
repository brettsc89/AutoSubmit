/* Code for calculating the ACF and CCF components from MD time series...
 *
 *
 * This code can be compiled like this: g++ xcorr2.cpp -lfftw3 -o cor -std=c++11
 * if the required fftw3 library has been installed. 
 *
 *Execute the code in the following way:
 * ./cor {file list name.txt} {output file name.txt}  {Number} 
 * where {file list name} is the name of your file that contains the ordered list of data files
 * to compute, {output file name.txt} is the name of your output file, and {Number} is the number of 
 * time steps to save in the autocorrelation (i.e. give the ACF from 0 to N*dt, where dt is the
 * size of the time step between data entries). 
 * For me this looks like: ./cor Jz_list.txt ACF.txt 10000   
Brett S Scheiner 6/11/19
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include<complex.h>
#include <limits>

typedef std::vector<double>doub_v;

//Defining class to handle time series data
class TimeSeries{

  doub_v series;
  const int  n = series.size();

public:
  TimeSeries(doub_v x): series(x){};   //Constructor for time series class
  doub_v ACF();                        //Function declaration for autocorrelation
  doub_v CCF( TimeSeries &series2);    //Function declaration for cross correlation
  doub_v getSeries(){return series;};  //Function definition for obtaining private doub_v series 
  double Variance();
  int    Length(){return n;};          // Returns the length of the series
};


// Function for calculating the ACF using the fftw3 library
doub_v TimeSeries::ACF() {
  doub_v fftout(n);  //double vector for the output of ACF

  //Note: that the fftw_complex type is not compatible with complex operations in std
  //The complex type is bit compatible, however, and the type of pointers to these variables can 
  //be reinterpreted as fftw_complex. Hence, the usage in the function call is 
  // reinterpret_cast<fftw_complex*>(out). 
  std::complex<double>* in  = new std::complex<double>[n];
  std::complex<double>* in2  = new std::complex<double>[n];
  std::complex<double>* out = new std::complex<double>[n];
  std::complex<double>* out2 = new std::complex<double>[n];
   
  //Setup fft plan for the forward transform of the TimeSeries and the reverse transform of 
  //its m odulus squared. See fftw3 documentation.
  fftw_plan pa = fftw_plan_dft_1d
    (n , reinterpret_cast<fftw_complex*>(in), reinterpret_cast<fftw_complex*>(out), 
     FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_plan pb = fftw_plan_dft_1d
    ( n , reinterpret_cast<fftw_complex*>(in2), reinterpret_cast<fftw_complex*>(out2),
      FFTW_BACKWARD, FFTW_ESTIMATE);


  //Initialization of fft input should occur after creation of the fft plan
  for(int i=0; i<n; i++)
    in[i]=series[i];

  //calculate the fft
  fftw_execute(pa);

  //Factor for proper scaling of the fft output
  double scale = 1.0/( n -1);

  //Calculate the modulus squared (power spectrum)
  for (int i = 0; i < n; i++)
    in2[i] = real(out[i] * conj(out[i])) * scale;
  
  //calculate the inverse fft
  fftw_execute(pb);

  //copy the real part to the output vector. Imag(out2) should be small.
  for (int i = 0; i < n; i++)
    fftout[i] =real( out2[i])* scale;

  
  //delete fft plan variable
  fftw_destroy_plan(pa);
  fftw_destroy_plan(pb);

  return fftout;

}


//Function for calculating the variance of the time series
double TimeSeries::Variance()
{
  
  double mean,V;  //variables for storing the mean and variance
  double varsum=0;


  for (int i(n); i > 0; --i)
    mean += series[i-1];

  mean=mean/n;

  for(int i(n); i>0; --i)
    varsum += std::pow(series[i]-mean,2.0);

  V=varsum/(n-1);

  return V;
}



//Function for calculating the cross correlation. See comments in ACF (similar to this function).
doub_v TimeSeries::CCF(TimeSeries &series2) {
  
  //Get the values of the argument time series. The values of the first time series are already
  // visible to class member functions. 
  doub_v fftout(n);
  doub_v seriesB(n);
  seriesB=series2.getSeries();


  std::complex<double>* inA  = new std::complex<double>[n];
  std::complex<double>* inB  = new std::complex<double>[n];
  std::complex<double>* inrFFT  = new std::complex<double>[n];
  std::complex<double>* outFFTA = new std::complex<double>[n];
  std::complex<double>* outFFTB = new std::complex<double>[n];
  std::complex<double>* outrFFT = new std::complex<double>[n];

  //FFT plan for the first time series
  fftw_plan pA = fftw_plan_dft_1d
    (n , reinterpret_cast<fftw_complex*>(inA), reinterpret_cast<fftw_complex*>(outFFTA), 
     FFTW_FORWARD, FFTW_ESTIMATE);
  //The ccf has an additional fft of the second time series
  fftw_plan pB = fftw_plan_dft_1d
    (n , reinterpret_cast<fftw_complex*>(inB), reinterpret_cast<fftw_complex*>(outFFTB), 
     FFTW_FORWARD, FFTW_ESTIMATE);
  //FFT plan for the reverse transform of the cross power spectrum
  fftw_plan pC = fftw_plan_dft_1d
    (n , reinterpret_cast<fftw_complex*>(inrFFT), reinterpret_cast<fftw_complex*>(outrFFT), 
     FFTW_BACKWARD, FFTW_ESTIMATE);


  //Initialize input values after fft plan definition
  for(int i=0; i<n; i++){
    inA[i]=series[i];
    inB[i]=seriesB[i];
  }

  fftw_execute(pA);
  fftw_execute(pB);

  
  double scale = 1.0/( n -1);
  
  //calculate the cross power spectrum
  for (int i = 0; i < n; i++)
    inrFFT[i] = real(outFFTA[i] * conj(outFFTB[i])) * scale;

  fftw_execute(pC);

  for (int i = 0; i < n; i++)
    fftout[i] =real( outrFFT[i])* scale;


  fftw_destroy_plan(pA);
  fftw_destroy_plan(pB);
  fftw_destroy_plan(pC);

  return fftout;
}


//==================================================//
//=================MAIN PROGRAM ====================//
//==================================================//

int main(int argc, char **argv)
{

  using namespace std;
  
  doub_v vecT, vecK, vecP, vecV, vecJ;
  double t, k, p, v;
  double var;
  ifstream inputFile;
  ofstream outputFile(argv[2]);
  vector<string> files;
  string line;
  int number = std::atoi (argv[3]);



  //open file with list of data files given as the first 
  //argument in the command line
  inputFile.open(argv[1]);

  //Read lines into vector of data file names
  while (getline(inputFile, line))   
  {

    files.push_back(line);
    cout<<line<<endl;

  }

  inputFile.close();

  //Read in the data from each file in files
  for(int i=0;i<files.size(); i++)
  {
    inputFile.open(files[i]);
    std::cout<<"Reading file "<<files[i]<<std::endl;


    // Ignore the first header line
    inputFile.ignore ( std::numeric_limits<std::streamsize>::max(), '\n' );
    //Read the remaining lines
    while (inputFile >> t >> k >> p >> v)
      {
	//Add the time series to data arrays
	vecT.push_back(t);      //time
	vecK.push_back(k);      //kinetic part
	vecP.push_back(p);      //potential part
	vecV.push_back(v);      //virial part
	vecJ.push_back(k+p+v);  //total 
      }

    inputFile.close();
    std::cout<<"Completed reading of file "<<files[i]<<std::endl;

  };



  //Create time series from the vectors read in above
  TimeSeries J(vecJ), P(vecP), K(vecK), V(vecV);
  doub_v ccfJJ, ccfKK, ccfPP, ccfVV, ccfPK, ccfKP, ccfPV, ccfVP, ccfKV, ccfVK;

  //calculate ACFs and CCFs from time series
  cout << "Computing JJ ACF"<<endl;
  ccfJJ = J.ACF();
  cout << "Computing KK ACF"<<endl;
  ccfKK = K.ACF();
  cout << "Computing PP ACF"<<endl;
  ccfPP = P.ACF();
  cout << "Computing VV ACF"<<endl;
  ccfVV = V.ACF();
  cout << "Computing PK CCF"<<endl;
  ccfPK = P.CCF(K);
  cout << "Computing KP CCF"<<endl;
  ccfKP = K.CCF(P);
  cout << "Computing PV CCF"<<endl;
  ccfPV = P.CCF(V);
  cout << "Computing VP CCF"<<endl;
  ccfVP = V.CCF(P);
  cout << "Computing KV CCF"<<endl;
  ccfKV = K.CCF(V);
  cout << "Computing VK CCF"<<endl;
  ccfVK = V.CCF(K);


  //Write a header for the output file
  outputFile << "vecT" << " , " << "ccfJJ"<< " , "<<"ccfKK"<<" , "<<"ccfPP"<<" , "<<"ccfVV"<<
	" , "<<"ccfPK"<<" , "<<"ccfKP"<<" , "<<"ccfPV"<<" , "<<"ccfVP"<<" , "<<"ccfKV"<<
	" , "<<"ccfVK"<<std::endl;

  //Write CCFs to file
  for(int i(0); i<number;i++)
    {
     
      outputFile << vecT[i] << " , " << ccfJJ[i]<< " , "<<ccfKK[i]<<" , "<<ccfPP[i]<<" , "<<ccfVV[i]<<
	" , "<<ccfPK[i]<<" , "<<ccfKP[i]<<" , "<<ccfPV[i]<<" , "<<ccfVP[i]<<" , "<<ccfKV[i]<<
	" , "<<ccfVK[i]<<std::endl;

    }

  
  return 0;
  
}
  
