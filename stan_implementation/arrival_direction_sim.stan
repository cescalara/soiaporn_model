/**
 * Simulation of UHECR arrival directions.
 * Based on the Auger observing periods reported in Abreu et al. 2010.
 * @author Francesca Capel
 * @date July 2018
 */

functions{
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
   * Sample point on sphere orthogonal to mu.
   */
  vector sample_orthonormal_to_rng(vector mu) {

    int dim = num_elements(mu);
    vector[dim] v;
    vector[dim] proj_mu_v;
    vector[dim] orthto;
    
    for (i in 1:dim) {
     v[i] = normal_rng(0, 1);
    }
    
    proj_mu_v = mu * dot_product(mu, v) / abs_val(mu);
    orthto = v - proj_mu_v;
    
    return (orthto / abs_val(orthto));

  }
  
  /**
   * Rejection sampling scheme for sampling distance from center on
   * surface of the sphere.
   */
  real sample_weight_rng(real kappa, int dim) {

    int sdim = dim - 1; /* as S^{n-1} */
    real b = sdim / (sqrt(4. * pow(kappa, 2) + pow(sdim, 2)) + 2 * kappa);
    real x = (1 - b) / (1 + b);
    real c = kappa * x + sdim * log(1 - pow(x, 2));

    int i = 0;
    real z;
    real w;
    real u;
    while (i == 0) {
      z = beta_rng(sdim / 2, sdim / 2);
      w = (1 - (1 + b) * z) / (1 - (1 - b) * z);
      u = uniform_rng(0, 1);
      if (kappa * w + sdim * log(1 - x * w) - c >= log(u)) {
	i = 1;
      }
    }

    return w;
  }
  
  /**
   * Generate an N-dimensional sample from the von Mises - Fisher
   * distribution around center mu in R^N with concentration kappa.
   */
  vector vMF_rng(vector mu, real kappa) {

    int dim = num_elements(mu);
    vector[dim] result;

    real w = sample_weight_rng(kappa, dim);
    vector[dim] v = sample_orthonormal_to_rng(mu);

    result = ( v * sqrt(1 - pow(w, 2)) ) + (w * mu);
    return result;
   
  }

  /**
   * Sample a point uniformly from the surface of a sphere of 
   * a certain radius.
   */
  vector sphere_rng(real radius) {

    vector[3] result;
    real u = uniform_rng(0, 1);
    real v = uniform_rng(0, 1);
    real theta = 2 * pi() * u;
    real phi = acos( (2 * v) - 1 );

    result[1] = radius * cos(theta) * sin(phi); 
    result[2] = radius * sin(theta) * sin(phi); 
    result[3] = radius * cos(phi);

    return result;
    
  }
  
  /**
   * Calculate weights from source distances.
   */
  vector get_source_weights(vector D) {

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
   * Calculate weights from exposure integral.
   */
  vector get_exposure_weights(vector F, vector eps, real alpha_T) {

    int N = num_elements(F);
    vector[N] weights;
    
    real normalisation = 0;

    for (i in 1:N-1) {
      normalisation += F[i] * eps[i];
    }
    normalisation += F[N] * (alpha_T / (4 * pi()));
    
    for (i in 1:N-1) {
      weights[i] = (F[i] * eps[i]) / normalisation;
    }
    weights[N] = (F[N] * (alpha_T / (4 * pi()))) / normalisation;
    
    return weights;
  }


  /**
   * Convert from unit vector omega to theta of spherical coordinate system.
   * @param omega a 3D unit vector.
   */
  real omega_to_theta(vector omega) {

    real theta;
    
    int N = num_elements(omega);

    if (N != 3) {
      print("Error: input vector omega must be of 3 dimensions");
    }

    theta = acos(omega[3]);
    
    return theta;
  }

  /**
   * Calculate xi part of exposure.
   * @param theta from 0 to pi.
   * @param p observatory dependent parameters.
   */
  real xi(real theta, real[] p) { 
    return (p[3] - (p[2] * cos(theta))) / (p[1] * sin(theta));
  }

  /**
   * Calculate alpha_m part of exposure.
   * @param theta from 0 to pi.
   * @param p observatory dependent parameters.
   */
  real alpha_m(real theta, real[] p) {

    real am;
    
    real xi_val = xi(theta, p);
    if (xi_val > 1) {
      am = 0;
    }
    else if (xi_val < -1) {
      am = pi();
    }
    else {
      am = acos(xi_val);
    }

    return am;
  }

  /**
   * Calculate the exposure factor for a given position on the sky. 
   * @param theta from 0 to pi.
   * @param p observatory dependent parameters.
   */
  real m(real theta, real[] p) {
    return (p[1] * sin(theta) * sin(alpha_m(theta, p)) 
            + alpha_m(theta, p) * p[2] * cos(theta));
  }

  real get_Nex_sim(vector F, vector eps, real alpha_T) {

    int N = num_elements(F);
    real Nex = 0;

    for (i in 1:N-1) {
      Nex += F[i] * eps[i];
    }
    Nex += F[N] * (alpha_T / (4 * pi()));

    return Nex;
  }
  
}

data {

  /* flux */
  real<lower=0> F_T;
  real <lower=0, upper=1> f;

  /* deflection */
  real<lower=0> kappa;
  real<lower=0> kappa_c;

  /* sources */
  int<lower=0> N_A;
  unit_vector[3] varpi[N_A];
  vector[N_A] D;

  /* observatory parameters */
  real A;
  real a_0;
  real<lower=0> theta_m;
  real<lower=0> alpha_T;
  vector[N_A] eps;
  
}

transformed data {

  simplex[N_A] w = get_source_weights(D);
  int<lower=0> N;
  real<lower=0> Nex;
  
  vector[N_A + 1] F;
  simplex[N_A + 1] w_exposure;
  
  real params[3];
  real m_max;
  
  for (k in 1:N_A) {
    F[k] = f * w[k] * F_T;
  }
  F[N_A + 1] = (1 - f) * F_T;

  w_exposure = get_exposure_weights(F, eps, alpha_T);
  Nex = get_Nex_sim(F, eps, alpha_T);
  print ("Nex:", Nex);
  
  N = poisson_rng(Nex);
  
  params[1] = cos(a_0);
  params[2] = sin(a_0);
  params[3] = cos(theta_m);
    
  m_max = m(pi(), params);
}

generated quantities {

  int lambda[N];
  unit_vector[3] omega;
  real theta[N];
  real pdet[N];
  real accept;
  simplex[2] p;
  unit_vector[3] event[N];
  real Nex_sim = Nex;
  
  for (i in 1:N) {
    
    lambda[i] = categorical_rng(w_exposure);

    /* source */
    if (lambda[i] < N_A + 1) {
      accept = 0;
      while (accept != 1) {
	omega = vMF_rng(varpi[lambda[i]], kappa);
	theta[i] = omega_to_theta(omega);
	pdet[i] = m(theta[i], params) / m_max;
	p[1] = pdet[i];
	p[2] = 1 - pdet[i];
	accept = categorical_rng(p);
      }
    }
    /* background */
    else {
      accept = 0;
      while (accept != 1) {
	omega = sphere_rng(1);
	theta[i] = omega_to_theta(omega);
	pdet[i] = m(theta[i], params) / m_max;
	p[1] = pdet[i];
	p[2] = 1 - pdet[i];
	accept = categorical_rng(p);
      }
    }

    event[i] = vMF_rng(omega, kappa_c);  	  
 
  }
  
}

