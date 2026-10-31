// The part of Boost.Math that OpenVINS uses: quantile(chi_squared(k), p), the gate of its chi-squared outlier tests
// (tabulated for k = 1..499 when the updaters are built). Inverts the regularized lower incomplete gamma function
// P(k/2, x/2) = p by safeguarded Newton steps; accurate to ~1e-12 relative.
#ifndef VIOBENCH_COMPAT_BOOST_CHI_SQUARED_HPP
#define VIOBENCH_COMPAT_BOOST_CHI_SQUARED_HPP

#include <cmath>
#include <limits>

namespace boost {
namespace math {

class chi_squared {
public:
  explicit chi_squared(double degrees_of_freedom = 1) : degrees_of_freedom_(degrees_of_freedom) {}
  double degrees_of_freedom() const { return degrees_of_freedom_; }

private:
  double degrees_of_freedom_;
};

namespace detail {

// P(a, x): series below a + 1, Lentz's continued fraction for Q(a, x) above (Numerical Recipes, ch. 6.2)
inline double regularized_gamma_p(double a, double x) {
  if (x <= 0)
    return 0;
  const double log_prefix = a * std::log(x) - x - std::lgamma(a);
  const double epsilon = 1e-15;
  if (x < a + 1) {
    double term = 1 / a, sum = term;
    for (int n = 1; n < 1000 && std::fabs(term) > std::fabs(sum) * epsilon; n++) {
      term *= x / (a + n);
      sum += term;
    }
    return sum * std::exp(log_prefix);
  }
  const double tiny = std::numeric_limits<double>::min() / epsilon;
  double b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
  for (int n = 1; n < 1000; n++) {
    const double an = -n * (n - a);
    b += 2;
    d = an * d + b;
    d = std::fabs(d) < tiny ? tiny : d;
    c = b + an / c;
    c = std::fabs(c) < tiny ? tiny : c;
    d = 1 / d;
    const double delta = d * c;
    h *= delta;
    if (std::fabs(delta - 1) < epsilon)
      break;
  }
  return 1 - std::exp(log_prefix) * h;
}

} // namespace detail

inline double quantile(const chi_squared &distribution, double p) {
  const double k = distribution.degrees_of_freedom();
  if (!(k > 0) || !(p > 0) || !(p < 1))
    return p <= 0 ? 0 : std::numeric_limits<double>::infinity();
  // Newton on F(x) = P(k/2, x/2) - p from the mean, inside a shrinking bracket
  double x = k, low = 0, high = std::numeric_limits<double>::infinity();
  const double a = 0.5 * k, log_normalizer = -a * std::log(2.0) - std::lgamma(a);
  for (int i = 0; i < 100; i++) {
    const double f = detail::regularized_gamma_p(a, 0.5 * x) - p;
    (f < 0 ? low : high) = x;
    const double density = std::exp(log_normalizer + (a - 1) * std::log(x) - 0.5 * x);
    double next = x - f / density;
    if (!(next > low && next < high)) // outside the bracket: bisect (doubling while there is no upper bound)
      next = std::isinf(high) ? 2 * x : 0.5 * (low + high);
    if (std::fabs(next - x) <= 1e-14 * x)
      return next;
    x = next;
  }
  return x;
}

} // namespace math
} // namespace boost

#endif
