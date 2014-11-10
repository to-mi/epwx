/*
 * Tomi Peltola, tomi.peltola@aalto.fi
 * http://becs.aalto.fi/en/research/bayes/epwx/
 */
#include <Eigen/Eigen>
#include <math.h>

using namespace std;
using namespace Eigen;

// in:  lik, Mu_cav_w, Prec_cav_w, Mu_cav_x, Prec_cav_x
// out: Mean_tilt_w, Cov_tilt_w, Mean_tilt_x, Cov_tilt_x, z
template <class T_Likelihood>
void compute(const Likelihood<T_Likelihood>& lik,
             const Ref<const VectorXd>& Mu_cav_w,
             const Ref<const MatrixXd>& Prec_cav_w,
             const Ref<const VectorXd>& Mu_cav_x,
             const Ref<const MatrixXd>& Prec_cav_x,
             Ref<VectorXd> Mean_tilt_w,
             Ref<MatrixXd> Cov_tilt_w,
             Ref<VectorXd> Mean_tilt_x,
             Ref<MatrixXd> Cov_tilt_x,
             double& z)
{
  // constants
  const double T = 30.0;

  // diagonalize prec_cav
  size_t K = Prec_cav_w.rows();

  EigenSolver<MatrixXd> es(Prec_cav_w * Prec_cav_x);
  const Ref<const MatrixXd> S_(es.eigenvectors().real());
  const Ref<const VectorXd> L_(es.eigenvalues().real());

  VectorXd L(2 * K);
  L.head(K) = L_.array().sqrt().inverse();
  L.tail(K) = -L.head(K);

  MatrixXd S2(S_);
  MatrixXd S1(K, K);
  for (int k = 0; k < K; ++k){
    S1.col(k) = Prec_cav_x * S2.col(k);
    double scale = sqrt(2 * (S2.col(k).dot(S1.col(k))));
    S2.col(k) /= scale;
    S1.col(k) *= L(k) / scale;
  }

  // get s
  const double s = lik.get_s(L);
  const double s2 = s*s;

  VectorXd v(2*K);
  {
    VectorXd tmp(S1.transpose() * Mu_cav_w);
    v.head(K) = S2.transpose() * Mu_cav_x;
    v.tail(K) = v.head(K) - tmp;
    v.head(K) += tmp;
  }

  const VectorXd v2(v.array().square()); // is .array() needed here?
  const VectorXd L2(L.array().square());
  const VectorXd one_m_sL(1.0 - s * L.array());

  // estimate integration end point...
  double tend2 = T / (0.25 * (L2.array() / one_m_sL.array().square()).sum()
                 + 0.5 * (v2.array() * L2.array() / one_m_sL.array().cube()).sum()
                 - lik.d_re_ll_at_zero(s2));

  // newton's iterations
  const double f_const_term = lik.re_ll_at_zero(s2) - 0.5 * one_m_sL.array().log().sum()
                              + 0.5 * (v2.array() / one_m_sL.array()).sum()
                              - T;
  for (int i = 0; i < 100; ++i){
    VectorXd denom(tend2 * L2.array() + one_m_sL.array().square());
    VectorXd tmp1(v2.array() * one_m_sL.array() / denom.array());
    double f_tend2 = f_const_term - lik.re_ll_at(tend2, s2)
                     + 0.25 * denom.array().log().sum()
                     - 0.5 * tmp1.array().sum();

    if (f_tend2 >= 0.0) break;

    VectorXd tmp2(L2.array() / denom.array());
    double df_tend2 = -lik.d_re_ll_at(tend2, s2)
                      + 0.25 * tmp2.array().sum()
                      + 0.5 * (tmp1.array() * tmp2.array()).sum();

    tend2 -= f_tend2 / df_tend2;
  }
  // TODO: check if iterations run out?
  const double tend = sqrt(tend2);

  // ...and number of evaluation points
  // Taylor approx for period (TODO: this does not seem to work very well)
  const double period = (2 * M_PI - lik.im_ll_at_zero(s)) 
                        / (lik.d_im_ll_at_zero(s) 
                           - 0.5 * (L.array() / one_m_sL.array() * (1 + v2.array() / one_m_sL.array())).sum());
  size_t neval = std::min(MAX_EVALS, std::max((int)std::ceil(tend / period * EVALS_PER_PERIOD), MIN_EVALS));
  neval += neval % 2;

  // normalization integrand
  VectorXd diag(2 * K);
  MatrixXd diagdiag(2 * K, 2 * K);

  double log_z_re_normalization = lik.re_ll_at_zero(s2)
                                  - 0.5 * one_m_sL.array().log().sum()
                                  + 0.5 * (v2.array() / one_m_sL.array()).sum();
  //log_z_re_zero = 0.0;
  double log_z_im_zero = lik.im_ll_at_zero(s);

  // zeroth evaluation:
  z = cos(log_z_im_zero);
  diag = one_m_sL.array().inverse();
  diagdiag = (one_m_sL * one_m_sL.transpose()).array().inverse().matrix();

  // 1..neval evaluations:
  for (int i = 1; i <= neval; ++i){
    const double t = tend * ((double)(i) / (double)(neval));
    const double t2 = t * t;
    const double w_simp = 2.0 + (2 * (i % 2) - (i == neval));

    VectorXcd D(one_m_sL.array().cast<complex<double> >() + L.array().cast<complex<double> >() * complex<double>(0.0, t));
    VectorXcd iD(D.array().inverse());
    complex<double> z_t = exp(0.5 * v2.dot(iD) 
                              -0.5 * D.array().log().sum() // is this ok?
                              +complex<double>(lik.re_ll_at(t2, s2) - log_z_re_normalization, lik.im_ll_at(t, s)))
                          * w_simp;

    z += z_t.real();

    diag.array() += iD.real().array() * z_t.real() - iD.imag().array() * z_t.imag();
    MatrixXcd diagdiag_tmp(iD * iD.transpose());
    diagdiag.array() += diagdiag_tmp.real().array() * z_t.real() - diagdiag_tmp.imag().array() * z_t.imag();
  }

  if (z <= 0) return; // skip update if z is clearly problematic

  diag /= z;
  diagdiag /= z;

  Mean_tilt_w = diag.head(K).asDiagonal() * v.head(K) - diag.tail(K).asDiagonal() * v.tail(K);
  Mean_tilt_w = (S1 * Mean_tilt_w).eval();
  Mean_tilt_x = diag.head(K).asDiagonal() * v.head(K) + diag.tail(K).asDiagonal() * v.tail(K);
  Mean_tilt_x = (S2 * Mean_tilt_x).eval();

  diagdiag = (v.asDiagonal() * diagdiag * v.asDiagonal()).eval();
  Cov_tilt_w = diagdiag.topLeftCorner(K,K);
  Cov_tilt_w += diagdiag.bottomRightCorner(K,K);
  Cov_tilt_w -= diagdiag.bottomLeftCorner(K,K);
  Cov_tilt_w -= diagdiag.topRightCorner(K,K);
  Cov_tilt_w.diagonal() += diag.head(K);
  Cov_tilt_w.diagonal() += diag.tail(K);
  Cov_tilt_w = (S1 * Cov_tilt_w * S1.transpose()).eval();
  Cov_tilt_w -= Mean_tilt_w * Mean_tilt_w.transpose();

  Cov_tilt_x = diagdiag.topLeftCorner(K,K);
  Cov_tilt_x += diagdiag.bottomRightCorner(K,K);
  Cov_tilt_x += diagdiag.bottomLeftCorner(K,K);
  Cov_tilt_x += diagdiag.topRightCorner(K,K);
  Cov_tilt_x.diagonal() += diag.head(K);
  Cov_tilt_x.diagonal() += diag.tail(K);
  Cov_tilt_x = (S2 * Cov_tilt_x * S2.transpose()).eval();
  Cov_tilt_x -= Mean_tilt_x * Mean_tilt_x.transpose();

  z *= tend / neval / 3.0;
}
