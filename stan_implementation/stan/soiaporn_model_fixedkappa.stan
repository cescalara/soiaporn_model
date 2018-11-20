/**
 * Model for UHECR arrival directions.
 * Model altered to include the Soiaporn et al. priors and fixed kappa.
 * @author Francesca Capel
 * @date July 2018
 */


functions {

  /**
   * compute the absolute value of a vector
   */
  real abs_val(vector input_vector) {
    real av;
    int n = num_elements(input_vector);

    real sum_squares = 0;
    for (i in 1:n) {
      sum_squares += (input_vector[i] * input_vector[i]);
    }
    av = sqrt(sum_squares);

    return av;
  }
  
  /**
   * Calculate weights from source distances.
   */
  vector get_weights(vector D) {

    int N = num_elements(D);
    vector[N] weights;
    
    real normalisation = 0;

    for (i in 1:N) {
      normalisation += (1 / pow(D[i], 2));
    }
    for (i in 1:N) {
      weights[i] = (1 / pow(D[i], 2)) / normalisation;
    }
      
    return weights;
  }
  
  /**
   * Interpolate x from a given set of x and y values.
   */
  real interpolate(vector x_values, vector y_values, real x) {
    real x_left;
    real y_left;
    real x_right;
    real y_right;
    real dydx;

    int Nx = num_elements(x_values);
    real xmin = x_values[1];
    real xmax = x_values[Nx];
    int i = 1;

    if (x > xmax || x < xmin) {
      print("Warning, x is outside of interpolation range!");
      print("Returning edge values.");
      print("x:", x);
      print("xmax", xmax);
      
      if(x > xmax) {
	return y_values[Nx];
      }
      else if (x < xmin) {
	return y_values[1];
      }
    }
    
    if( x >= x_values[Nx - 1] ) {
      i = Nx - 1;
    }
    else {
      while( x > x_values[i + 1] ) { i = i+1; }
    }

    x_left = x_values[i];
    y_left = y_values[i];
    x_right = x_values[i + 1];
    y_right = y_values[i + 1];

    dydx = (y_right - y_left) / (x_right - x_left);
    
    return y_left + dydx * (x - x_left);
  }

  /**
   * Calculate the N_ex for a given kappa by
   * interpolating over a vector of eps values
   * for each source.
   */
  real get_Nex(vector F, vector[] eps, vector kappa_grid, real kappa, real alpha_T) {

    real eps_from_kappa;
    int N = num_elements(F);
    real Nex = 0;

    for (i in 1:N-1) {
      eps_from_kappa = interpolate(kappa_grid, eps[i], kappa);
      Nex += F[i] * eps_from_kappa;
    }
    Nex += F[N] * (alpha_T / (4 * pi()));

    return Nex;
  }
  
  /**
   * Define the fik PDF.
   * NB: Cannot be vectorised.
   * Uses sinh(kappa) ~ exp(kappa)/2 
   * approximation for kappa > 100.
   */
  real fik_lpdf(vector v, vector mu, real kappa, real kappa_c) {

    real lprob;
    real inner = abs_val((kappa_c * v) + (kappa * mu));
    
    if (kappa > 100 || kappa_c > 100) {
      lprob = log(kappa * kappa_c) - log(4 * pi() * inner) + inner - (kappa + kappa_c) + log(2);
    }
    else {   
      lprob = log(kappa * kappa_c) - log(4 * pi() * sinh(kappa) * sinh(kappa_c)) + log(sinh(inner)) - log(inner);
    }

    return lprob;   
  }
  
}

data {

  /* sources */
  int<lower=0> N_A;
  unit_vector[3] varpi[N_A]; 
  vector[N_A] D;
  
  /* uhecr */
  int<lower=0> N; 
  unit_vector[3] detected[N]; 
  vector[N] zenith_angle;
  vector[N] A;
  
  /* observatory */
  real<lower=100, upper=10000> kappa_c;  
  real<lower=0> alpha_T;
  int Ngrid;
  vector[Ngrid] eps[N_A];
  vector[Ngrid] kappa_grid;

  /* deflection */
  real kappa;
  
}

transformed data {

  /* known weights on D */
  simplex[N_A] w = get_weights(D);

}

parameters { 

  /* sources */
  real<lower=0> F_T; 

  /* associated fraction */
  real<lower=0, upper=1> f; 
  
}

transformed parameters {

  vector[N_A + 1] F;

  for (k in 1:N_A) {
    F[k] = f * w[k] * F_T;
  }
  F[N_A + 1] = (1 - f) * F_T;

}

model {

  vector[N_A + 1] log_F;
  real Nex;
  
  log_F = log(F);

  /* Nex */
  Nex = get_Nex(F, eps, kappa_grid, kappa, alpha_T);

  /* rate factor */
  for (i in 1:N) {

    vector[N_A + 1] lps = log_F;

    for (k in 1:N_A + 1) {
      if (k < N_A + 1) {
	lps[k] += fik_lpdf(detected[i] | varpi[k], kappa, kappa_c);
      }
      else {
	lps[k] += log(1 / ( 4 * pi() ));
      }
      
      lps[k] += log(A[i] * zenith_angle[i]);
    }
    target += log_sum_exp(lps);
  }
  
  /* normalise */
  target += -Nex; 

  /* priors */
  F_T ~ exponential(0.01 * 4 * pi()); 
  f ~ beta(1, 1);
  
}
