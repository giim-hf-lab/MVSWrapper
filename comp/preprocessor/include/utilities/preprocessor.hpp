#ifndef __UTILITIES_PREPROCESSOR_HPP__
#define __UTILITIES_PREPROCESSOR_HPP__

#include <type_traits>

#if __cplusplus >= 201402L
#  define UTILITIES_ENABLE_IF(B, T) std::enable_if_t<B, T>
#elif __cplusplus >= 201103L
#  define UTILITIES_ENABLE_IF(B, T) typename std::enable_if<B, T>::type
#endif


#if defined(__cpp_concepts)
#  define UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(N, C) template<typename T> requires C<T>
#elif defined(UTILITIES_ENABLE_IF)
#  define UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(N, C) template<typename T, const UTILITIES_ENABLE_IF(C<N>, T) * = nullptr>
#endif

#endif
