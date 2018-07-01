#ifndef _METROPOLIS_WITHIN_GIBBS_H
#define _METROPOLIS_WITHIN_GIBBS_H

#include <cmath>
#include <random>
#include <vector>

/* define min/max values for parameters */
#define F_T_MIN 0.0
#define F_T_MAX 0.5
#define f_MIN 0.0
#define f_MAX 1.0

/**
 * Pass input parameters to the MetropolisWithinGibbs object.
 */
typedef struct InputParameters {

  /* values of fixed parameters */
  double kappa;
  double kappa_c;

  /* hyperparametrs */
  int a;
  int b;
  real s;
  
  /* constants */
  double alpha_T;
  double M;

};

/**
 * Pass input data to the MetropolisWithinGibbs object.
 */
typedef struct InputData {

  /* uhecr */
  std::vector<std::array<double, 3>> d;
  std::vector<double> theta;
  
  /* sources */
  std::vector<std::array<double, 3>> varpi;
  std::vector<double> D;

  /* integral table */
  std::vector<double> eps;
  
};


typedef struct Samples {

  std::vector<double> F_T;
  std::vector<double> f;

};

/**
 * Manage Metropolis-within-Gibbs sampling of the Soiaporn et al. 2012 model
 * for UHECR arrival directions
 * @author Francesca Capel
 * @date July 2018
 */
class MetropolisWithinGibbs {
  
 public:
  std::vector<Samples> samples; 

  MetropolisWithinGibbs(InputParameters input, InputData data);
  int Sample(int Niter, int Nchain);
  
private:
  InputParameters input_parameters;
  InputData input_data;
  int N_C;
  
  /* random number generation */
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<double> F_T_init_dist(F_T_MIN, F_T_MAX);
  std::uniform_real_distribution<double> f_init_dist(f_MIN, f_MAX);

};

#endif /* _METROPOLIS_WITHIN_GIBBS_H */
