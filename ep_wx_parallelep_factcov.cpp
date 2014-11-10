/*
 * Tomi Peltola, tomi.peltola@aalto.fi
 * http://becs.aalto.fi/en/research/bayes/epwx/
 *
 * Compile in Matlab with "mex -largeArrayDims -I/path/to/eigen/ ep_wx_parallelep_factcov.cpp"
 */
#include <Eigen/Eigen>
#include <iostream>
#include "mex.h"

// quadrature settings (TODO: make user configurable?)
#define MAX_EVALS 5120
#define MIN_EVALS 128
#define EVALS_PER_PERIOD 11

// likelihood coding
enum LikelihoodCoding { GAUSSIAN = 0, PROBIT = 1 };

using namespace Eigen;
typedef Map<MatrixXd> MexMat;
typedef Map<VectorXd> MexVec;

#include "likelihoods.hpp"
#include "compute_tilt_factor_covariance.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // prhs: N, M, K, fa, si, z, y, y_lik_type, prior, opts
  const size_t N = (size_t)mxGetScalar(prhs[0]);
  const size_t M = (size_t)mxGetScalar(prhs[1]);
  const size_t K = (size_t)mxGetScalar(prhs[2]);
  const size_t K2 = K*K;

  double *const fa_w_Mu_pr = mxGetPr(mxGetField(mxGetField(prhs[3], 0, "w"), 0, "Mu"));
  double *const fa_w_Tau_pr = mxGetPr(mxGetField(mxGetField(prhs[3], 0, "w"), 0, "Tau"));
  double *const fa_x_Mu_pr = mxGetPr(mxGetField(mxGetField(prhs[3], 0, "x"), 0, "Mu"));
  double *const fa_x_Tau_pr = mxGetPr(mxGetField(mxGetField(prhs[3], 0, "x"), 0, "Tau"));

  double *const si_w_Mu_pr = mxGetPr(mxGetField(mxGetField(prhs[4], 0, "w"), 0, "Mu"));
  double *const si_w_Tau_pr = mxGetPr(mxGetField(mxGetField(prhs[4], 0, "w"), 0, "Tau"));
  double *const si_x_Mu_pr = mxGetPr(mxGetField(mxGetField(prhs[4], 0, "x"), 0, "Mu"));
  double *const si_x_Tau_pr = mxGetPr(mxGetField(mxGetField(prhs[4], 0, "x"), 0, "Tau"));
  double * z = mxGetPr(prhs[5]);

  const MexMat ys(mxGetPr(prhs[6]), N, M);
  const MexVec ys_likelihood_type(mxGetPr(prhs[7]), M);

  const mxArray *const prior = prhs[8];

  const double damp = mxGetScalar(mxGetField(prhs[9], 0, "damp"));
  const double frac = mxGetScalar(mxGetField(prhs[9], 0, "frac"));
  const double one_m_damp = 1.0 - damp;
  const double min_site_prec = mxGetScalar(mxGetField(prhs[9], 0, "min_site_prec"));

  // likelihood site updates
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < M; ++j){
      const MexMat fa_x_Tau(fa_x_Tau_pr + i * K2, K, K);
      const MexVec fa_x_Mu(fa_x_Mu_pr + i * K, K);
      const MexMat fa_w_Tau(fa_w_Tau_pr + j * K2, K, K);
      const MexVec fa_w_Mu(fa_w_Mu_pr + j * K, K);
      
      MexMat si_x_Tau(si_x_Tau_pr + (j * N + i) * K2, K, K);
      MexVec si_x_Mu(si_x_Mu_pr + (j * N + i) * K, K);
      MexMat si_w_Tau(si_w_Tau_pr + (i * M + j) * K2, K, K);
      MexVec si_w_Mu(si_w_Mu_pr + (i * M + j) * K, K);


      // compute cavity
      MatrixXd Tau_cav_w(fa_w_Tau - frac * si_w_Tau);
      VectorXd Mu_cav_w(fa_w_Mu - frac * si_w_Mu);
      MatrixXd Tau_cav_x(fa_x_Tau - frac * si_x_Tau);
      VectorXd Mu_cav_x(fa_x_Mu - frac * si_x_Mu);

      // compute tilt
      MatrixXd Cov_tilt_w(K, K);
      MatrixXd Cov_tilt_x(K, K);
      VectorXd Mean_tilt_w(K);
      VectorXd Mean_tilt_x(K);
      switch ((int)ys_likelihood_type(j)) {
        case GAUSSIAN:
          {
            const Likelihood<Gaussian> lik(ys(i, j), mxGetScalar(mxGetField(prior, 0, "sigma2")), frac);
            compute(lik, Mu_cav_w, Tau_cav_w, Mu_cav_x, Tau_cav_x, Mean_tilt_w, Cov_tilt_w, Mean_tilt_x, Cov_tilt_x, *z);
          }
          break;
        case PROBIT:
          {
            const Likelihood<Probit> lik(ys(i, j), 0.5);
            compute(lik, Mu_cav_w, Tau_cav_w, Mu_cav_x, Tau_cav_x, Mean_tilt_w, Cov_tilt_w, Mean_tilt_x, Cov_tilt_x, *z);
          }
          break;
      }

      if (std::isnan(*z) || (*z <= 1e-8)){
        ++z;
        continue; // skip update
        // TODO: better heuristics for identifying poor numerical integrals
      }
      ++z;

      // update sites
      MatrixXd Tau_tilt_w(Cov_tilt_w.inverse());
      Tau_tilt_w = (0.5 * (Tau_tilt_w + Tau_tilt_w.transpose())).eval(); // TODO: is this necessary/useful?

      MatrixXd new_Tau_w = one_m_damp * si_w_Tau + (damp / frac) * (Tau_tilt_w - Tau_cav_w);
      {
        bool changed = false;
        SelfAdjointEigenSolver<MatrixXd> eig(new_Tau_w);
        VectorXd eigv(eig.eigenvalues());
          
        for (size_t ii = 0; ii < eigv.size(); ++ii) {
          if (eigv(ii) < min_site_prec) {
              eigv(ii) = min_site_prec;
              changed = true;
          }
        }
        if (changed) {
          new_Tau_w = eig.eigenvectors() * eigv.asDiagonal() * eig.eigenvectors().transpose();
          // it seems that the above does not retain symmetry and the error may accumulate, so ensure it
          new_Tau_w = (0.5 * (new_Tau_w + new_Tau_w.transpose())).eval();
          Tau_tilt_w = (frac / damp) * new_Tau_w - (one_m_damp * frac / damp) * si_w_Tau + Tau_cav_w;
        }
      }

      si_w_Tau = new_Tau_w;
      si_w_Mu = (one_m_damp * si_w_Mu + (damp / frac) * (Tau_tilt_w * Mean_tilt_w - Mu_cav_w)).eval();

      MatrixXd Tau_tilt_x(Cov_tilt_x.inverse());
      Tau_tilt_x = (0.5 * (Tau_tilt_x + Tau_tilt_x.transpose())).eval(); // TODO: is this necessary/useful?

      MatrixXd new_Tau_x = one_m_damp * si_x_Tau + (damp / frac) * (Tau_tilt_x - Tau_cav_x);
      {
        bool changed = false;
        SelfAdjointEigenSolver<MatrixXd> eig(new_Tau_x);
        VectorXd eigv(eig.eigenvalues());
          
        for (size_t ii = 0; ii < eigv.size(); ++ii) {
          if (eigv(ii) < min_site_prec) {
              eigv(ii) = min_site_prec;
              changed = true;
          }
        }
        if (changed) {
          new_Tau_x = eig.eigenvectors() * eigv.asDiagonal() * eig.eigenvectors().transpose();
          // it seems that the above does not retain symmetry and the error may accumulate, so ensure it
          new_Tau_x = (0.5 * (new_Tau_x + new_Tau_x.transpose())).eval();
          Tau_tilt_x = (frac / damp) * new_Tau_x - (one_m_damp * frac / damp) * si_x_Tau + Tau_cav_x;
        }
      }

      si_x_Tau = new_Tau_x;
      si_x_Mu = (one_m_damp * si_x_Mu + (damp / frac) * (Tau_tilt_x * Mean_tilt_x - Mu_cav_x)).eval();
    }
  }

  return;
}
