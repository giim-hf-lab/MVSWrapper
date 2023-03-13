#ifndef __UTILITIES_PREPROCESSOR_HPP__
#define __UTILITIES_PREPROCESSOR_HPP__

#if __cpp_lib_concepts >= 202002L
#  include <concepts>
#endif
#include <type_traits>

#ifdef _MSC_VER
#  if _UTILITIES_MSVC_SAL
#    include <sal.h>
#  endif
#else
#  undef _UTILITIES_MSVC_SAL
#endif

#if __cplusplus >= 201103L
#  define _UTILITIES_CPP11_FEATURES 1
#  if __cplusplus >= 201402L
#    define _UTILITIES_CPP14_FEATURES 1
#    if __cplusplus >= 201703L
#      define _UTILITIES_CPP17_FEATURES 1
#      if __cplusplus >= 202002L
#        define _UTILITIES_CPP20_FEATURES 1
#      endif
#    endif
#  endif
#endif

#if _UTILITIES_CPP14_FEATURES
#  define UTILITIES_DEPRECATED [[deprecated]]
#  define UTILITIES_DEPRECATED_REASON(R) [[deprecated(R)]]
#elif defined(_MSC_VER)
#  define UTILITIES_DEPRECATED __declspec(deprecated)
#  define UTILITIES_DEPRECATED_REASON(R) __declspec(deprecated(R))
#elif defined(__GNUC__) || defined(__clang__)
#  define UTILITIES_DEPRECATED __attribute__((deprecated))
#  define UTILITIES_DEPRECATED_REASON(R) __attribute__((deprecated(R)))
#else
#  define _UTILITIES_TRIVIAL_DEPRECATED 1
#  define UTILITIES_DEPRECATED
#  define UTILITIES_DEPRECATED_REASON(R)
#endif

#if _UTILITIES_MSVC_SAL
#  define UTILITIES_NODISCARD _Check_return_
#elif _UTILITIES_CPP17_FEATURES
#  define UTILITIES_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#  define UTILITIES_NODISCARD __attribute__((warn_unused_result))
#else
#  define _UTILITIES_TRIVIAL_NODISCARD 1
#  define UTILITIES_NODISCARD
#endif

#if __cpp_concepts >= 201907L
#  define UTILITIES_FUNCTION_TEMPLATE(R) template<typename F, typename... Args> requires std::is_invocable_r_v<R, F, Args...>
#elif _UTILITIES_CPP17_FEATURES
#  define UTILITIES_FUNCTION_TEMPLATE(R) template<typename F, typename... Args, typename = std::enable_if_t<std::is_invocable_r_v<R, F, Args...>>>
#elif _UTILITIES_CPP11_FEATURES
#  define _UTILITIES_TRIVIAL_FUNCTION_TEMPLATE 1
#  define UTILITIES_FUNCTION_TEMPLATE(R) template<typename F, typename... Args>
#endif

#if _UTILITIES_CPP14_FEATURES
#  define UTILITIES_ENABLE_IF(B) std::enable_if_t<B>
#  define UTILITIES_ENABLE_IF_T(B, T) std::enable_if_t<B, T>
#elif _UTILITIES_CPP11_FEATURES
#  define UTILITIES_ENABLE_IF(B) typename std::enable_if<B>::type
#  define UTILITIES_ENABLE_IF_T(B, T) typename std::enable_if<B, T>::type
#endif

#if __cpp_concepts >= 201907L
#  define UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(N, C) template<typename T> requires C<T>
#elif defined(UTILITIES_ENABLE_IF)
#  define UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(N, C) template<typename T, const UTILITIES_ENABLE_IF_T(C<N>, T) * = nullptr>
#endif

#if __cpp_lib_concepts >= 202002L
#  define UTILITIES_SIMPLE_CONCEPT_TEMPLATE(N, NS, C) template<NS##C N>
#elif defined(UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT)
#  define UTILITIES_SIMPLE_CONCEPT_TEMPLATE(N, NS, C) UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(N, NS##is_##C##_v)
#endif

#endif
