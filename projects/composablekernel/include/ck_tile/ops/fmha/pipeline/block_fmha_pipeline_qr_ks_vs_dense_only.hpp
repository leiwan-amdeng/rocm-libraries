// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
template <typename Problem_,
          typename Policy_         = BlockFmhaPipelineQRKSVSDefaultPolicy,
          bool PaddedVecLoadStore_ = false>
struct BlockFmhaPipelineQRKSVSDenseOnly
{
    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    // Public aliases required by FmhaFwdKernel argument plumbing. DenseOnly does not use
    // the corresponding feature paths internally.
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using QScaleDataType        = remove_cvref_t<typename Problem::QScaleDataType>;
    using KScaleDataType        = remove_cvref_t<typename Problem::KScaleDataType>;
    using VScaleDataType        = remove_cvref_t<typename Problem::VScaleDataType>;
    using PScaleDataType        = remove_cvref_t<typename Problem::PScaleDataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;

    template <typename T>
    using has_partial_k_support = decltype(T::kSupportsPartialK);
    template <typename T>
    using has_partial_n_support = decltype(T::kSupportsPartialN);

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode        = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ         = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK         = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ        = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV        = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap   = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum            = Problem::BiasEnum;
    static constexpr bool kStoreLSE           = Problem::kStoreLSE;
    static constexpr bool kHasDropout         = Problem::kHasDropout;
    static constexpr bool kHasSink            = Problem::kHasSink;
    static constexpr bool kPaddedVecLoadStore = PaddedVecLoadStore_;
    static constexpr bool kUseHdimTailArgs    = kPadHeadDimQ || kPadHeadDimV;



    static constexpr uint32_t DS_READ = 0x100; // Barrier for DS (data share) read
    static constexpr uint32_t MFMA    = 0x008; // Barrier for MFMA (matrix multiply-accumulate)

    static_assert(!kPaddedVecLoadStore || (kPadHeadDimQ && kPadHeadDimV),
                  "padded vector load/store fast path only applies to padded head-dim kernels");

    // Dense/no-feature specialization for the measured gfx1201 bf16 d128 path.
    // The generated instance still uses compile-time seq padding (pssk), so keep seq-pad
    // handling, but reject head padding/mask/bias/dropout/LSE/sink variants.
    static_assert(!kPadHeadDimQ && !kPadHeadDimV);
    static_assert(!kHasLogitsSoftCap && !kStoreLSE && !kHasDropout && !kHasSink);
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
    static_assert(!FmhaMask::IsMasking);
    static_assert(!kPaddedVecLoadStore);

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = (kPadHeadDimQ && !kPaddedVecLoadStore)
                                               ? numeric_traits<QDataType>::PackedSize
                                               : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = (kPadHeadDimQ && !kPaddedVecLoadStore)
                                               ? numeric_traits<KDataType>::PackedSize
                                               : Policy::template GetAlignmentK<Problem>();
    // DenseOnly targets V row-major, full head dim, no sequence masking/padding.
    static constexpr index_t kAlignmentV = Policy::template GetAlignmentV<Problem>();

    static constexpr index_t kAlignmentO =
        (kPadHeadDimV && !kPaddedVecLoadStore) ? 1 : Policy::template GetAlignmentO<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim <= 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            }
        }
    }();

    static constexpr const char* name = "qr_dense_only";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices,
              typename QScaleDramBlockWindowTmp,
              typename KScaleDramBlockWindowTmp,
              typename VScaleDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const float* k_descale_ptr,
               const float* v_descale_ptr,
               const index_t block_scale_size_kv,
               const QScaleDramBlockWindowTmp&
                   q_scale_dram_block_window_tmp, // M0*(K0/kQKScaleGranularity) tile
               const KScaleDramBlockWindowTmp&
                   k_scale_dram_block_window_tmp, // N0*(K0/kQKScaleGranularity) tile
               const VScaleDramBlockWindowTmp&
                   v_scale_dram_block_window_tmp, // N1*(K1/kVScaleGranularity) tile
               const float sink_v,
               const index_t valid_k0_loops,
               const index_t valid_last_k0_length,
               const index_t valid_n1_length) const
    {
        (void)bias_dram_block_window_tmp;
        (void)bias_element_func;
        (void)randval_dram_block_window_tmp;
        (void)lse_dram_window_tmp;
        (void)lse_element_func;
        (void)dropout;
        (void)k_descale_ptr;
        (void)v_descale_ptr;
        (void)block_scale_size_kv;
        (void)v_element_func;
        (void)p_compute_element_func;
        (void)q_scale_dram_block_window_tmp;
        (void)k_scale_dram_block_window_tmp;
        (void)v_scale_dram_block_window_tmp;
        (void)sink_v;
        (void)valid_k0_loops;
        (void)valid_last_k0_length;
        (void)valid_n1_length;
        (void)position_encoding;
        (void)variant;
        (void)variant_params;
        (void)block_indices;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kSubQKHeaddim ==
                              QDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // K tile in LDS
        KDataType* k_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQ<Problem>()));
        auto k_lds           = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0                      = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1                      = Policy::template GetKVBlockGemm<Problem>();
        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        auto q = load_tile(q_dram_window);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);
        const auto q_origin = q_dram_window.get_window_origin();

        const auto tile_range_result = [&mask, &q_origin]() {
            auto [start, end] =
                mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
            return ck_tile::make_tuple(0, start, end);
        }();
        const auto sink_seq_end   = tile_range_result.get(ck_tile::number<0>{});
        const auto seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto kv_load_start = (sink_seq_end == 0 && seqlen_k_start > 0) ? seqlen_k_start : 0;
        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);
        const auto num_total_loop =
            integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0) + num_sink_loop;

        // Dense-only: no mask/pad early-exit path.

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {kv_load_start, 0});

        // Dense-only: no bias/dropout random-value windows.

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, kv_load_start}, // TODO: hdim split?
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto q_tile = tile_elementwise_in(q_element_func, q);

        // prefetch K tile
        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;
        // Number of k0 iterations prefetched ahead of the current compute iteration.
        constexpr index_t kK0PrefetchDepth = 2;
        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);
        do
        {
            // Dense-only: no K/V blockscale descale.
            // STAGE 1, QK gemm
            auto k_dram_window = make_tile_window(
                k_dram_block_window.get_bottom_tensor_view(),
                k_dram_block_window.get_window_lengths(),
                k_dram_block_window.get_window_origin(),
                Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                        // load
            // Dense-only: no K scale tile load.

            clear_tile(s_acc); // initialize C
            // Dense-only: no bias tile load.

            auto run_gemm_0 = [&](auto i_k0) {
                // Dense-only: no padded head-dim partial-K tail.

                auto q_slice = get_slice_tile(
                    q_tile, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{});
                gemm_0(s_acc, q_slice, k_lds_window);
            };

            auto v_prefetch = decltype(load_tile(v_dram_window)){};
            // gfx12 dense-only path: issue V prefetch after QK GEMM tail (baseline-good).
            if constexpr(k0_loops == 1)
            {
                auto k_block_tile = load_tile(k_dram_window);
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                block_sync_lds();
                run_gemm_0(number<0>{});
            }
            else
            {
                static_assert(kK0PrefetchDepth <= k0_loops);
                auto k_block_tile = load_tile(k_dram_window);
                move_tile_window(k_dram_window, {0, kK0});
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                k_block_tile = load_tile(k_dram_window);

                if constexpr(k0_loops > kK0PrefetchDepth)
                {
                    static_for<0, k0_loops - kK0PrefetchDepth, 1>{}([&](auto i_k0) {
                        block_sync_lds();
                        run_gemm_0(number<i_k0>{});
                        block_sync_lds();
                        move_tile_window(k_dram_window, {0, kK0});

                        store_tile(
                            k_lds_window,
                            tile_elementwise_in(k_element_func, k_block_tile)); // LDS write i + 1
                        k_block_tile = load_tile(k_dram_window);                // global read i + 2
                    });
                }

                { // tail
                    block_sync_lds();
                    run_gemm_0(number<k0_loops - kK0PrefetchDepth>{});
                    block_sync_lds();

                    store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));

                    block_sync_lds();

                    run_gemm_0(number<k0_loops - 1>{});
                }
            }
            load_tile(v_prefetch, v_dram_window);
            // STAGE 2, scale_s, add bias, mask, softmax
            s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
            // Dense-only full-sequence/no-mask path: no seq-k padding or masking.
            const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution()); // Pcompute{j}

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) { return raw_m; };

            // Conditional rescaling: skip o_acc rescale when correction factor
            // exp2(acc_scale_log2) is negligible (< exp2(-8) ~= 0.004, below BF16
            // precision). Adapted from FlashAttention-4 (Tri Dao, 2025).
            // Eliminates 70-90% of rescale operations in practice.
            //
            // For skip rows we stabilize P with m_old (the previous max) instead of
            // the new max m_j, so P is computed directly in the m_{j-1} frame and no
            // post-correction sweep is needed. For rescale rows we use m_j as usual.
            // FP8 quant modes (PERTENSOR/BLOCKSCALE/etc.) cast P to FP8 after
            // softmax. In the skip branch P is computed with m_old, so P can
            // exceed the FP8 representable range and saturate, corrupting the
            // P*V GEMM. Disable skip for all FP8 paths (threshold 0).
            static constexpr SMPLComputeDataType kRescaleThreshold =
                type_convert<SMPLComputeDataType>(
5.0f);

            // Per-row stabilizer: m_old for skip rows, m_j for rescale rows.
            auto m_stab =
                make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
            // Per-row rescale factor (exp2 of acc_scale_log2); only valid when
            // needs_rescale[i] is true.
            auto rescale_factor =
                make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
            auto needs_rescale = make_static_distributed_tensor<bool>(m.get_tile_distribution());
            set_tile(needs_rescale, false);

            constexpr auto m_spans = decltype(m)::get_distributed_spans();
            sweep_tile_span(m_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto row_max        = scale_s * get_validated_m(m[i_idx]);
                const auto acc_scale_log2 = scale_s * m_old[i_idx] - row_max;
                const bool need_rescale   =
                    (acc_scale_log2 < type_convert<SMPLComputeDataType>(-kRescaleThreshold));

                if(need_rescale)
                {
                    rescale_factor(i_idx) = exp2(acc_scale_log2);
                    m_stab(i_idx)         = m[i_idx];
                    needs_rescale(i_idx)  = true;
                }
                else
                {
                    // Skip branch: stabilize P with m_old so P is already in
                    // m_{j-1} frame; restore m to m_old for downstream iterations.
                    m_stab(i_idx) = m_old[i_idx];
                    m(i_idx)      = m_old[i_idx];
                }
            });

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_max         = scale_s * get_validated_m(m_stab[i_idx]);
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p_compute(i_j_idx)     = exp2(scale_s * s[i_j_idx] - row_max);
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                if(needs_rescale[i_idx])
                {
                    const auto tmp = rescale_factor[i_idx];
                    l(i_idx)       = tmp * l[i_idx] + rowsum_p[i_idx];
                    sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= tmp;
                    });
                }
                else
                {
                    // Skip: P already in m_{j-1} frame, no o_acc rescale needed.
                    l(i_idx) = l[i_idx] + rowsum_p[i_idx];
                }
            });

            // Dense-only: no dropout.

            block_sync_lds();
            auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
            shuffle_tile(v_shuffle_tmp, v_prefetch);
            store_tile(v_lds_window, v_shuffle_tmp); // store the prefetch
            // store_tile(v_lds_window, v_prefetch); // store the prefetch


            move_tile_window(v_dram_window, {0, kK1});

            const auto p = cast_tile<PDataType>(p_compute);

            // STAGE 3, KV gemm
            // Dense-only: accumulate PV directly into o_acc.

            auto run_gemm_1_impl =
                [&](auto& o_acc_tensor, const auto& p_slice, const auto&... gemm_1_args) {
                    gemm_1(o_acc_tensor, p_slice, gemm_1_args...);
                };

            auto run_gemm_1 = [&](auto i_k1) {
                auto p_slice =
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{});
                run_gemm_1_impl(o_acc, p_slice, v_lds_window);
            };

            // Dense-only d_v=128 with k1_loops=2: always run the pre-tail PV stage.
            static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                const auto v = load_tile(v_dram_window); // load next v
                block_sync_lds();
                run_gemm_1(number<i_k1>{});
                block_sync_lds();
                auto v_shuffle_next = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_next, v);
                store_tile(v_lds_window, v_shuffle_next); // store next v

                //store_tile(v_lds_window, v); // store next v

                move_tile_window(v_dram_window, {0, kK1});
            });

            // move K tile windows; DenseOnly has no sink path.
            move_tile_window(k_dram_block_window, {kN0, 0});
            // Dense-only: no K scale window movement.
            // tail
            {
                block_sync_lds();
                run_gemm_1(number<k1_loops - 1>{});
                block_sync_lds();
            }
            // Dense-only: no blockscale O accumulation merge.
        } while(++i_total_loops < num_total_loop);

        // Dense-only: no LSE store.

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = 1 / l[i_idx];
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        // DenseOnly no-output-transform path: o_acc_element_func is identity.
        (void)o_acc_element_func;

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices,
              typename QScaleDramBlockWindowTmp,
              typename KScaleDramBlockWindowTmp,
              typename VScaleDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const float* k_descale_ptr,
               const float* v_descale_ptr,
               const index_t block_scale_size_kv,
               const QScaleDramBlockWindowTmp&
                   q_scale_dram_block_window_tmp, // M0*(K0/kQKScaleGranularity) tile
               const KScaleDramBlockWindowTmp&
                   k_scale_dram_block_window_tmp, // N0*(K0/kQKScaleGranularity) tile
               const VScaleDramBlockWindowTmp&
                   v_scale_dram_block_window_tmp, // N1*(K1/kVScaleGranularity) tile
               const float sink_v) const
    {
        return operator()(q_dram_block_window_tmp,
                          q_element_func,
                          k_dram_block_window_tmp,
                          k_element_func,
                          v_dram_block_window_tmp,
                          v_element_func,
                          bias_dram_block_window_tmp,
                          bias_element_func,
                          randval_dram_block_window_tmp,
                          lse_dram_window_tmp,
                          lse_element_func,
                          s_acc_element_func,
                          p_compute_element_func,
                          o_acc_element_func,
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          dropout,
                          k_descale_ptr,
                          v_descale_ptr,
                          block_scale_size_kv,
                          q_scale_dram_block_window_tmp,
                          k_scale_dram_block_window_tmp,
                          v_scale_dram_block_window_tmp,
                          sink_v,
                          kQKHeaddim / kK0,
                          kK0,
                          kN1);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const float sink_v,
               const index_t valid_k0_loops,
               const index_t valid_last_k0_length,
               const index_t valid_n1_length) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          dropout,
                          nullptr,
                          nullptr,
                          1,
                          make_null_tile_window(make_tuple()),
                          make_null_tile_window(make_tuple()),
                          make_null_tile_window(make_tuple()),
                          sink_v,
                          valid_k0_loops,
                          valid_last_k0_length,
                          valid_n1_length);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const float sink_v) const
    {
        return operator()(q_dram_block_window_tmp,
                          k_dram_block_window_tmp,
                          v_dram_block_window_tmp,
                          bias_dram_block_window_tmp,
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          dropout,
                          sink_v,
                          kQKHeaddim / kK0,
                          kK0,
                          kN1);
    }
};

template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSDefaultPolicy>
using BlockFmhaPipelineQRKSVSDenseOnlyHpad = BlockFmhaPipelineQRKSVSDenseOnly<Problem_, Policy_, true>;

} // namespace ck_tile
