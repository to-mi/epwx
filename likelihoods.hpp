/*
 * Tomi Peltola, tomi.peltola@aalto.fi
 * http://becs.aalto.fi/en/research/bayes/epwx/
 */
#include <Eigen/Eigen>

using namespace Eigen;

class Gaussian;
class Probit;
template<class T_Likelihood> class Likelihood;

template<>
class Likelihood<Gaussian>
{
  public:
    const double y;
    const double neg_half_sigma2_div_frac;

    Likelihood(const double y, const double sigma2, const double frac)
      : y(y), neg_half_sigma2_div_frac(-0.5 * sigma2 / frac)
    {}

    // real part of log-likelihood at zero
    double re_ll_at_zero(const double s2) const {
      return 0.0;
    }

    // imag part of log-likelihood at zero
    double im_ll_at_zero(const double s) const {
      return 0.0;
    }

    // real part of log-likelihood at t^2
    double re_ll_at(const double t2, const double s2) const {
      return neg_half_sigma2_div_frac * t2;
    }

    // imag part of log-likelihood at t
    double im_ll_at(const double t, const double s) const {
      return y * t;
    }

    // derivative (wrt t^2) of real part of log-likelihood at zero
    double d_re_ll_at_zero(const double s2) const {
      return neg_half_sigma2_div_frac;
    }

    // derivative (wrt t) of imag part of log-likelihood at zero
    double d_im_ll_at_zero(const double s) const {
      return y;
    }

    // derivative (wrt t^2) of real part of log-likelihood at t^2
    double d_re_ll_at(const double t2, const double s2) const {
      return neg_half_sigma2_div_frac;
    }

    // derivative (wrt t) of imag part of log-likelihood at t
    double d_im_ll_at(const double t, const double s) const {
      return y;
    }

    // ...
    double get_s(const Ref<const VectorXd>& eigenvalues) const {
      return 0.0;
    }
};

// Probit likelihood does not support fraction parameter
template<>
class Likelihood<Probit>
{
  public:
    const double y;
    const double q; // used in choosing s, should be in (0, 1)

    Likelihood(const double y, const double q)
      : y(y), q(q)
    {}

    // real part of log-likelihood at zero
    double re_ll_at_zero(const double s2) const {
      return -0.5 * (log(s2) - s2);
    }

    // imag part of log-likelihood at zero
    double im_ll_at_zero(const double s) const {
      return 0.0;
    }

    // real part of log-likelihood at t2
    double re_ll_at(const double t2, const double s2) const {
      return -0.5 * (t2 - s2 + log(t2 + s2));
    }

    // imag part of log-likelihood at t
    double im_ll_at(const double t, const double s) const {
      return atan(t/s) - t * s;
    }

    // derivative of real part of log-likelihood at zero
    double d_re_ll_at_zero(const double s2) const {
      return -0.5 * (1.0 + 1.0 / s2);
    }

    // derivative of imag part of log-likelihood at zero
    double d_im_ll_at_zero(const double s) const {
      return 1.0 / s - s;
    }

    // derivative of real part of log-likelihood at t2
    double d_re_ll_at(const double t2, const double s2) const {
      return -0.5 * (1.0 + 1.0 / (t2 + s2));
    }

    // derivative of imag part of log-likelihood at t
    double d_im_ll_at(const double t, const double s) const {
      return 1.0 / (s * (1.0 + t*t/(s*s))) - s;
    }

    // ...
    double get_s(const Ref<const VectorXd>& eigenvalues) const {
      if (y > 0) return std::min(0.5, q / eigenvalues.maxCoeff());
      else       return std::max(-0.5, q / eigenvalues.minCoeff());
    }
};
