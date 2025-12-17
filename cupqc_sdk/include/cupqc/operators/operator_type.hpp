// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUPQCDX_OPERATORS_OPERATOR_TYPE_HPP
#define CUPQCDX_OPERATORS_OPERATOR_TYPE_HPP

#include "commondx/detail/expressions.hpp"

namespace cupqc {
    enum class operator_type
    {
        algorithm,
        capacity,
        field,
        function,
        precision,
        security_category,
        size,
        sm,
        width,
        // execution
        block,
        thread,
        warp,
        // block only
        block_dim,
        batches_per_block,
   };
} // namespace cupqc

#endif // CUPQCDX_OPERATORS_OPERATOR_TYPE_HPP

