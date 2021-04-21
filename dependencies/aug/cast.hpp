#pragma once


namespace aug::cast
{
    constexpr auto ignore(const auto value) noexcept -> void
    {
        static_cast<void>(value);
    }
}