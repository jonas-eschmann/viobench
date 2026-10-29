// The part of Boost.Filesystem that OpenVINS uses (exists, remove, path, create_directories) is std::filesystem.
#ifndef RELPOSE_COMPAT_BOOST_FILESYSTEM_HPP
#define RELPOSE_COMPAT_BOOST_FILESYSTEM_HPP

#include <filesystem>

namespace boost {
namespace filesystem = std::filesystem;
} // namespace boost

#endif
