// The part of Boost.DateTime that OpenVINS uses (timing printouts): ptime, microsec_clock::local_time() and
// time_duration::total_microseconds(), on std::chrono. Nothing depends on these values but log lines. In the wasm module
// the clock stands still: fuel is the measure there, and OpenVINS formats some timing lines even when it does not print
// them, which would make the fuel of a step vary with the wall clock.
#ifndef VIOBENCH_COMPAT_BOOST_POSIX_TIME_HPP
#define VIOBENCH_COMPAT_BOOST_POSIX_TIME_HPP

#include <chrono>
#include <cstdint>

namespace boost {
namespace posix_time {

class time_duration {
public:
  explicit time_duration(std::chrono::steady_clock::duration duration = {}) : duration_(duration) {}
  int64_t total_microseconds() const { return std::chrono::duration_cast<std::chrono::microseconds>(duration_).count(); }
  int64_t total_milliseconds() const { return std::chrono::duration_cast<std::chrono::milliseconds>(duration_).count(); }

private:
  std::chrono::steady_clock::duration duration_;
};

class ptime {
public:
  ptime() = default;
  explicit ptime(std::chrono::steady_clock::time_point time) : time_(time) {}
  time_duration operator-(const ptime &other) const { return time_duration(time_ - other.time_); }

private:
  std::chrono::steady_clock::time_point time_{};
};

struct microsec_clock {
  static ptime local_time() {
#if defined(__wasi__)
    return ptime();
#else
    return ptime(std::chrono::steady_clock::now());
#endif
  }
  static ptime universal_time() { return local_time(); }
};

} // namespace posix_time
} // namespace boost

#endif
