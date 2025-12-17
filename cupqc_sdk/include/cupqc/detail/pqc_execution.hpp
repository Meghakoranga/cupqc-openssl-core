// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUPQC_DETAIL_PQC_EXECUTION_HPP
#define CUPQC_DETAIL_PQC_EXECUTION_HPP

#include "pqc_description.hpp"
#include "database.hpp"
#include "cuhash_types.hpp"
#include <cstdint>
#include <variant>

namespace cupqc {
    namespace detail {

        template<class... Operators>
        class pqc_execution: public pqc_full_description<Operators...>, public commondx::detail::execution_description_expression
        {
            using base_type = pqc_full_description<Operators...>;
            using this_type = pqc_execution<Operators...>;

        protected:

            /// ---- Constraints

            // We need Block operator to be specified exactly once
            static constexpr bool has_one_block = has_at_most_one_of<operator_type::block, this_type>::value;
            static_assert(has_one_block, "Can't create pqc function with two execution operators");
        };


        template<class... Operators>
        class pqc_block_execution: public pqc_execution<Operators...>
        {

            using this_type = pqc_block_execution<Operators...>;
            using base_type = pqc_execution<Operators...>;

            /// ---- Traits

            // Block Dimension
            // * Default value: selected by implementation
            static constexpr bool has_block_dim        = has_operator<operator_type::block_dim, base_type>::value;
            using default_pqc_block_dim                = BlockDim<128>;
            using this_pqc_block_dim                   = get_or_default_t<operator_type::block_dim, base_type, default_pqc_block_dim>;
            static constexpr auto this_pqc_block_dim_v = this_pqc_block_dim::value;

            // Batches per Block
            // * Default value: 1
            static constexpr bool has_batches_per_block        = has_operator<operator_type::batches_per_block, base_type>::value;
            using default_pqc_batches_per_block                = BatchesPerBlock<1>;
            using this_pqc_batches_per_block                   = get_or_default_t<operator_type::batches_per_block, base_type, default_pqc_batches_per_block>;
            static constexpr auto this_pqc_batches_per_block_v = this_pqc_batches_per_block::value;

            // Size  
            // * Default value: 2048
            static constexpr bool has_size    = has_operator<operator_type::size, base_type>::value;
            using default_pqc_size            = Size<2048>;
            using this_pqc_size               = get_or_default_t<operator_type::size, base_type, default_pqc_size>;
            static constexpr auto this_size_v = this_pqc_size::value;

            /// ---- Constraints
            static_assert(is_kem_algorithm(this_type::this_pqc_algorithm_v) || is_dss_algorithm(this_type::this_pqc_algorithm_v) || is_merkle_algorithm(this_type::this_pqc_algorithm_v), 
                            "Block execution is not supported for hashing.");
            static_assert(this_pqc_block_dim::y == 1 && this_pqc_block_dim::z == 1,
                          "Provided block dimension is invalid, y and z dimensions must both be 1.");
            static constexpr bool valid_block_dim = this_pqc_block_dim::flat_size >= 32 && this_pqc_block_dim::flat_size <= 1024;
            static_assert(valid_block_dim,
                          "Provided block dimension is invalid, BlockDim<> must have at least 32 threads, and can't have more than 1024 threads.");

            static_assert(this_pqc_batches_per_block_v > 0, "Providing number of batches per block is invalid.  BatchesPerBlock<0> is unsupported.");

            template<class this_t, function func, class T = void>
            using function_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<this_t::is_complete && this_t::this_pqc_function_v == func, T>;
            template<class this_t, function func, class T = void>
            using merkle_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<this_t::is_complete_merkle && this_t::this_pqc_function_v == func, T>;

            template<class this_t, algorithm alg, class T = void>
            using algorithm_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<this_t::is_complete && this_t::this_pqc_algorithm_v == alg, T>;

            /// ---- Accessors
        public:
            static constexpr auto BlockDim = this_pqc_block_dim_v;
            static constexpr auto SecurityCategory = base_type::this_pqc_security_category_v;
            static constexpr auto Size = this_size_v;

            /// ---- Execution
        public:
            static constexpr size_t workspace_size = database::global_memory_size<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_function_v>();
            static constexpr size_t shared_memory_size = database::shared_memory_size<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_function_v>();
            static constexpr size_t entropy_size = database::entropy_size<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_function_v>();

            // keygen
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* public_key, uint8_t* secret_key,
                                           uint8_t* entropy,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Keygen> {
                database::keygen<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(public_key, secret_key, entropy, workspace, smem_workspace);
            }

            // encaps
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* cipher_text, uint8_t* shared_secret, const uint8_t* public_key,
                                           uint8_t* entropy,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Encaps> {
                database::encaps<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(cipher_text, shared_secret, public_key, entropy, workspace, smem_workspace);
            }

            // decaps
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* shared_secret, const uint8_t* ciphertext, const uint8_t* secret_key,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Decaps> {
                database::decaps<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(shared_secret, ciphertext, secret_key, workspace, smem_workspace);
            }

            // sign
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* signature, const uint8_t* message, const size_t message_length, const uint8_t* secret_key,
                                           uint8_t* entropy, uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Sign> {

                constexpr uint8_t* context = nullptr;
                constexpr uint8_t context_length = 0;
                this->execute(signature, message, message_length, context, context_length, secret_key, entropy, workspace, smem_workspace);
            }
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* signature, const uint8_t* message, const size_t message_length, const uint8_t* context, const uint8_t context_length, const uint8_t* secret_key,
                                           uint8_t* entropy, uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Sign> {
                database::sign<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(signature, message, message_length, context, context_length, secret_key, entropy, workspace, smem_workspace);
            }

            // verify
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(const uint8_t* message, const size_t message_length, const uint8_t* signature, const uint8_t* public_key,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Verify, bool> {
                constexpr uint8_t* context = nullptr;
                constexpr uint8_t context_length = 0;
                return this->execute(message, message_length, context, context_length, signature, public_key, workspace, smem_workspace);
            }
            template<class this_t = this_type>
            inline __device__ auto execute(const uint8_t* message, const size_t message_length, const uint8_t* context, const uint8_t context_length, const uint8_t* signature, const uint8_t* public_key,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Verify, bool> {
                return database::verify<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(message, message_length, context, context_length, signature, public_key, workspace, smem_workspace);
            }

            // Merkle Tree operations
            template<class this_t = this_type, class Hash>
            inline __device__ auto create_leaf(typename this_type::this_pqc_precision_t* leaf, const typename this_type::this_pqc_precision_t* data, Hash& hash, const size_t inbuf_len)
                -> merkle_enable_if_t<this_t, function::Merkle> {
                return database::create_leaf<Hash, this_type::this_pqc_precision_t>(leaf, data, hash, inbuf_len);
            }

            template<class this_t = this_type, class Hash>
            inline __device__ auto generate_tree(Hash& hash, tree<this_type::this_size_v, Hash, typename this_type::this_pqc_precision_t>& merkle)
                -> merkle_enable_if_t<this_t, function::Merkle> {
                return database::generate_tree<this_type::this_size_v, Hash, this_type::this_pqc_precision_t>(hash, merkle);
            }

            template<class this_t = this_type, class Hash, uint32_t N>
            inline __device__ auto generate_sub_tree(Hash& hash, tree<N, Hash, typename this_type::this_pqc_precision_t>& merkle, const size_t sub_tree_num)
                -> merkle_enable_if_t<this_t, function::Merkle> {
                return database::generate_sub_tree<N, this_type::this_size_v, Hash, this_type::this_pqc_precision_t>(hash, merkle, sub_tree_num);
            }

            template<class this_t = this_type, class Hash>
            inline __device__ auto generate_proof(proof<this_type::this_size_v, Hash, typename this_type::this_pqc_precision_t>& proof,
                                                  const typename this_type::this_pqc_precision_t* proof_leaf, const uint32_t leaf_index,
                                                  const tree<this_type::this_size_v, Hash, typename this_type::this_pqc_precision_t>& merkle)
                -> merkle_enable_if_t<this_t, function::Merkle> {
                return database::generate_proof<this_type::this_size_v, Hash, this_type::this_pqc_precision_t>(proof, proof_leaf, leaf_index, merkle);
            }

            template<class this_t = this_type, class Hash>
            inline __device__ auto verify_proof(const proof<this_type::this_size_v, Hash, typename this_type::this_pqc_precision_t>& proof,
                                                const typename this_type::this_pqc_precision_t* verify_leaf, const uint32_t leaf_index,
                                                const typename this_type::this_pqc_precision_t* root, Hash& hash)
                -> merkle_enable_if_t<this_t, function::Merkle, bool> {
                return database::verify_proof<this_type::this_size_v, Hash, this_type::this_pqc_precision_t>(proof, verify_leaf, leaf_index, root, hash);
            }
        };


        template <typename Exec, algorithm Alg, typename T = void>
        struct pqc_hash_context_helper;
        
        template <typename Exec, typename T>
        struct pqc_hash_context_helper<Exec, algorithm::SHA3, T>
        {
            using type = typename database::KeccakContext<Exec, T::capacity>;
        };
        template <typename Exec, typename T>
        struct pqc_hash_context_helper<Exec, algorithm::SHAKE, T>
        {
            using type = typename database::KeccakContext<Exec, T::capacity>;
        };
        template <typename Exec, typename T>
        struct pqc_hash_context_helper<Exec, algorithm::SHA2_32, T>
        {
            using type = typename database::SHA2Context<Exec, uint32_t, T::digest>;
        };
        template <typename Exec, typename T>
        struct pqc_hash_context_helper<Exec, algorithm::SHA2_64, T>
        {
            using type = typename database::SHA2Context<Exec, uint64_t, T::digest>;
        };

        template <typename Exec, typename T>
        struct pqc_hash_context_helper<Exec, algorithm::POSEIDON2, T>
        {
            using type = typename database::Poseidon2Context<Exec, T::capacity, T::width, typename T::field_type>;
        };


        template<class... Operators>
        class pqc_thread_execution: public pqc_execution<Operators...>
        {

            using this_type = pqc_thread_execution<Operators...>;
            using base_type = pqc_execution<Operators...>;

            /// ---- Traits

            template<class this_t, function func, class T = void>
            using function_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<(this_t::is_complete || (this_t::is_complete_without_security_category && this_t::this_pqc_algorithm_v == algorithm::POSEIDON2)) 
                                                                              && this_t::this_pqc_function_v == func, T>;

            template<class this_t, function func, class T = void>
            using sha_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<this_t::is_complete  && this_t::this_pqc_algorithm_v != algorithm::POSEIDON2 && this_t::this_pqc_function_v == func, T>;

            template<class this_t, function func, class T = void>
            using poseidon2_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<this_t::is_complete_without_security_category  && this_t::this_pqc_algorithm_v == algorithm::POSEIDON2 && this_t::this_pqc_function_v == func, T>;

            using context_t = typename pqc_hash_context_helper<cupqc::Thread, this_type::this_pqc_algorithm_v, this_type>::type;
            
            [[no_unique_address]]
            context_t context;

            /// ---- Constraints
            static_assert(is_hash_algorithm(this_type::this_pqc_algorithm_v), 
                "Thread execution is only supported for hashing.");

            /// ---- Accessors
        public:
            static constexpr auto SecurityCategory = base_type::this_pqc_security_category_v;
            static constexpr bool has_security_category = base_type::has_security_category;
            /// ---- Execution
        public:

            // hash
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto reset(void)
                -> function_enable_if_t<this_t, function::Hash> {
                    context.reset();
            }

            template<class this_t = this_type>
            inline __device__ auto update(const uint8_t* buffer, size_t len)
                -> sha_enable_if_t<this_t, function::Hash> {
                    context.update(buffer, len);
            }
            template<class this_t = this_type>
            inline __device__ auto update(const uint32_t* buffer, size_t len)
                -> poseidon2_enable_if_t<this_t, function::Hash> {
                    context.update(buffer, len);
            }

            template<class this_t = this_type>
            inline __device__ auto finalize(void)
                -> function_enable_if_t<this_t, function::Hash> {
                    if constexpr (this_type::this_pqc_algorithm_v == algorithm::SHA2_32 || this_type::this_pqc_algorithm_v == algorithm::SHA2_64 || this_type::this_pqc_algorithm_v == algorithm::POSEIDON2) {
                        context.finalize();
                    } else {
                        context.finalize(this_type::pad);
                    }
            }

            template<class this_t = this_type>
            inline __device__ auto digest(uint8_t* buffer, size_t len)
                -> sha_enable_if_t<this_t, function::Hash> {
                    context.digest(buffer, len);
            }
            template<class this_t = this_type>
            inline __device__ auto digest(uint32_t* buffer, size_t len)
                -> poseidon2_enable_if_t<this_t, function::Hash> {
                    context.digest(buffer, len);
            }
        };

        template<class... Operators>
        class pqc_warp_execution: public pqc_execution<Operators...>
        {

            using this_type = pqc_warp_execution<Operators...>;
            using base_type = pqc_execution<Operators...>;

            /// ---- Traits
            template<class this_t, function func, class T = void>
            using function_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<this_t::is_complete && this_t::this_pqc_function_v == func, T>;

            using context_t = typename pqc_hash_context_helper<cupqc::Warp, this_type::this_pqc_algorithm_v, this_type>::type;

            [[no_unique_address]]
            context_t context;

            /// ---- Constraints
            static_assert(is_hash_algorithm(this_type::this_pqc_algorithm_v), 
                "Warp execution is only supported for hashing.");

            /// ---- Accessors
        public:

            /// ---- Execution
        public:

            // hash
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto reset(void)
                -> function_enable_if_t<this_t, function::Hash> {
                    context.reset();
            }
            template<class this_t = this_type>
            inline __device__ auto update(const uint8_t* buffer, size_t len)
                -> function_enable_if_t<this_t, function::Hash> {
                    context.update(buffer, len);
            }
            template<class this_t = this_type>
            inline __device__ auto finalize(void)
                -> function_enable_if_t<this_t, function::Hash> {
                    context.finalize(this_type::pad);
            }
            template<class this_t = this_type>
            inline __device__ auto digest(uint8_t* buffer, size_t len)
                -> function_enable_if_t<this_t, function::Hash> {
                    context.digest(buffer, len);
            }

        };

        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool has_block_operator     = has_operator<operator_type::block, pqc_operator_wrapper<Operators...>>::value;
            static constexpr bool has_thread_operator    = has_operator<operator_type::thread, pqc_operator_wrapper<Operators...>>::value;
            static constexpr bool has_warp_operator    = has_operator<operator_type::warp, pqc_operator_wrapper<Operators...>>::value;
            static constexpr bool has_execution_operator = has_block_operator || has_thread_operator || has_warp_operator;

            // TODO cuBLASDx conditionally instantiates this, check if cuPQCDx also needs conditional instantiation
            using execution_type = COMMONDX_STL_NAMESPACE::conditional_t<has_block_operator, pqc_block_execution<Operators...>, 
                                    COMMONDX_STL_NAMESPACE::conditional_t<has_thread_operator, pqc_thread_execution<Operators...>, 
                                     pqc_warp_execution<Operators...>>>;
            using description_type = pqc_full_description<Operators...>;

        public:
            using type = COMMONDX_STL_NAMESPACE::conditional_t<has_execution_operator, execution_type, description_type>;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;

    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
                                   detail::make_description_t<Operator1, Operator2>>::type {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::pqc_full_description<Operators1...>&,
                                                       const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator2>::value,
                                   detail::make_description_t<Operators1..., Operator2>>::type {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::pqc_full_description<Operators2...>&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator1>::value,
                                   detail::make_description_t<Operator1, Operators2...>>::type {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::pqc_full_description<Operators1...>&,
                                                       const detail::pqc_full_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cupqc

#endif // CUPQC_DETAIL_PQC_EXECUTION_HPP

