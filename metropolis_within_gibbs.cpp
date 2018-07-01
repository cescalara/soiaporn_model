#include "metropolis_within_gibbs.h"

/**
 * Constructor.
 */
MetropolisWithinGibbs::MetropolisWithinGibbs(InputParameters input_parameters, InputData input_data) {
  this->input_parameters = input_parameters;
  this->input_data = input_data;
  this->N_C = input_data.theta.size();
}

/**
 * Run the sampler.
 * @param Niter number of iterations per chain
 * @param Nchain number of chains
 */
int MetropolisWithinGibbs::Sample(int Niter, int Nchain) {

  /* initialise free parameters randomly */
  double F_T_init[Nchain];
  double f_init[Nchain];

  for (int i = 0; i < Nchain; i++) {
    F_T_init[i] = this->F_T_init_dist(this->engine);
    f_init[i] = this->f_init_dist(this->engine);
  }

  /* run chains */
  for (int i = 0; i < Nchain; i++) {
    chain = RunSampler(Niter, F_T_init[i], f_init[i]);
    this->samples.push_back(chain);
  }
  
  return 0;
}

/**
 * Run a single chain.
 * @param Niter number of iterations
 * @param F_T_init F_T initial value
 * @param f_init f initial value
 */
Samples MetropolisWithinGibbs::RunChain(int Niter, double F_T_init, double f_init) {

  chain_samples = Samples();

  s = this->input_parameters.s;
  f = f_init;
  eps = this->input_data.eps;
  w = GetWeights(this->input_data.D);
  
  for (int i = 0; i < Niter; i++) {

    /* F_T */
    std::gamma_distribution<double> F_T_dist(this->N_C + 1, scale_F_T(s, f, eps, w));
    F_T = F_T_dist(this->engine);
    chain_samples.F_T.push_back(F_T);

    /* lambda */
    int lambda[N_C];
    

  }
  
  return 0;
}


double scale_F_T(double s, double f, std::vector<double> eps, std::vector<double> w) {

  double sum_term = 0;
  vec_len = eps.size();

  for (int k = 0; k < vec_len; k++) { 
    sum_term += w[k] * eps[k];
  }
  
  double denom = (1 + s) + ((1 - f) * (this->input_parameters.alpha_T / 4 * pi())) + (f * sum_term);
  return (1 / denom);

}

std::vector<double> MetropolisWithinGibbs::GetWeights(std::vector<double> D) {

  std::vector<double> w;
  double norm = 0;

  for (auto &d : D) {
    norm += (1 / pow(d, 2))
  }
    
  for (auto &d : D) {
    w.push_back((1 / pow(d, 2)) / norm);
  }
  
  return w;
}
