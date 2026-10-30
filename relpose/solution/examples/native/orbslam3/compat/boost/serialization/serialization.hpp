// Declarations only: upstream's uninstantiated archive templates mention these.
// This build excludes serialization and cannot silently serialize empty data.
#pragma once
#include <cstddef>
namespace boost::serialization {
class access;
template<class T> struct array_wrapper { T* data; std::size_t size; };
template<class T> array_wrapper<T> make_array(T*, std::size_t);
template<class Base, class Derived> Base& base_object(Derived&);
}
