#pragma once

#include <type_traits>
#include <functional>
#include <tuple>
#include "cast.hpp"


namespace aug::outparameter
{
    template<typename Out>
    [[nodiscard]] constexpr auto handle(const auto invokable)
        noexcept(std::is_nothrow_invocable_v<decltype(invokable)>)
        requires(!std::is_void_v<decltype(predicate(std::declval<Out&>()))>)
    {
        Out out;
        auto returned {invokable(out)};
        return std::make_pair(out, returned);
    }

    template<typename Out>
    [[nodiscard]] constexpr auto handle(const auto predicate)
        noexcept(std::is_nothrow_invocable_v<decltype(predicate)>)
    {
        Out out;
        predicate(out);
        return out;
    }

    // returns only the outparameter and ignores the non-void return type
    template<typename Out>
    [[nodiscard]] constexpr auto handle_ignore(const auto invokable)
        noexcept(std::is_nothrow_invocable_v<decltype(invokable)>)
        requires(std::is_void_v<decltype(invokable(std::declval<Out&>()))>)
    {
        Out out;
        aug::cast::ignore(invokable(out));
        return out;
    }
}

#define OUT OUTPARAMETER_PLACEHOLDER
#define OUTPARAMATER_BINDING(function_call) [&](auto& OUTPARAMETER_PLACEHOLDER) { function_call; }
#define OUTPARAMETER(function_call, value_type) \
    ::aug::outparameter::handle<value_type>(OUTPARAMATER_BINDING(function_call))