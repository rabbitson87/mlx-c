/* lumen-rs Phase 1.8 M4.8 — C ABI for the lumen::flash_attn_bf16 mlx
 * primitive. Returns a lazy mlx_array; the kernel encodes into mlx's own
 * compute encoder when the array is evaluated.                          */

#ifndef MLX_LUMEN_H
#define MLX_LUMEN_H

#include <stdbool.h>
#include <stdint.h>

#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Build a lazy mlx_array for the lumen flash-attention kernel.
 *
 *  q     : [B, H,    Sq,  D]  bfloat16, D must be 256
 *  k     : [B, H_kv, Skv, D]  bfloat16, H % H_kv == 0
 *  v     : same shape as k    bfloat16
 *  mask  : [Sq, Skv]          bfloat16 (additive). Bind a null-ctx
 *          mlx_array (i.e. `mlx_array a; a.ctx = NULL;`) when has_mask=0.
 *  scale : scalar applied to QK^T before softmax.
 *
 * Returns 0 on success, 1 on mlx-level error (call mlx_error to retrieve).
 */
int mlx_lumen_flash_attn_bf16(
    mlx_array* result,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    int has_mask,
    mlx_array mask,
    float scale,
    mlx_stream s);

/**
 * FA-2 prefill (Sq>1). Causal + optional sliding window mask in-register.
 *
 *  q     : [B, H,    Sq,  D]  bfloat16, D ∈ {256, 512}, Sq > 1
 *  k, v  : [B, H_kv, Skv, D]  bfloat16
 *  scale : QK^T multiplier
 *  window_size : 0 = full causal; >0 = sliding window of given size.
 *                D=512 requires window_size=0.
 *  kv_offset   : absolute index of Q[0] in the K/V stream
 *
 * Returns 0 on success, 1 on mlx-level error.
 */
int mlx_lumen_flash_attn_prefill_bf16(
    mlx_array* result,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    float scale,
    uint32_t window_size,
    uint32_t kv_offset,
    mlx_stream s);

/**
 * Windowed scaled-dot-product attention — wraps
 * `mlx::core::fast::scaled_dot_product_attention` with `mask_mode="causal"`
 * and `window_size>0`, enabling the steel kernel's kb_start truncation +
 * left-edge masking. The kernel skips entire K-blocks below the window's
 * lower bound, yielding ~(L−W)/L compute savings for Sq=L>>W (e.g. 87.5%
 * skipped at L=8192, W=1024).
 *
 *  q, k, v : [B, H_q/H_kv, L, D]  any supported dtype (bf16/f16/f32)
 *  scale   : QK^T multiplier
 *  window_size : sliding window width (must be > 0; W=0 → use the standard
 *                `mlx_fast_scaled_dot_product_attention` API instead)
 *
 * Returns 0 on success.
 */
int mlx_lumen_sdpa_windowed(
    mlx_array* result,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    float scale,
    int window_size,
    mlx_stream s);

/**
 * TurboQuant Lloyd-Max nearest-centroid Stage-1 encode.
 *
 *  x_norm     : [..., D]  bfloat16, pre-normalized (≈ N(0, 1))
 *  boundaries : [n_inner = n_levels - 1]  float32  (inner Lloyd-Max boundaries)
 *  Output     : same shape as x_norm, uint8 codes in [0, n_levels - 1]
 *
 * Returns 0 on success.
 */
int mlx_lumen_turboquant_encode(
    mlx_array* result,
    mlx_array x_norm,
    mlx_array boundaries,
    mlx_stream s);

/**
 * QJL Stage-2 bit-pack: sign-bit of each element packed into u32 words.
 *
 *  values  : [..., m]  float32 (sign of each element drives the bit)
 *  m       : projection width (1..4096)
 *  Output  : [..., ceil(m/32)] uint32, bit j of word w = 1 iff
 *            values[..., w*32 + j] >= 0; bits past m are zero-padded.
 *
 * Returns 0 on success.
 */
int mlx_lumen_qjl_pack_signs(
    mlx_array* result,
    mlx_array values,
    int m,
    mlx_stream s);

/**
 * QJL Stage-2 bit-unpack: u32 sign-bits → bf16 ±1.
 *
 *  packed : [..., ceil(m/32)] uint32
 *  m      : projection width (1..4096)
 *  Output : [..., m]  bfloat16, values in {-1, +1}
 *
 * Returns 0 on success.
 */
int mlx_lumen_qjl_unpack_signs(
    mlx_array* result,
    mlx_array packed,
    int m,
    mlx_stream s);

/**
 * TurboQuant Stage-1 fused: σ + normalize + Lloyd-Max nearest-centroid
 * encode in a single kernel.
 *
 *  x_rot      : [..., D]  bfloat16, post-rotation but pre-σ-normalize.
 *               Last-axis D must be a positive multiple of 32, ≤ 1024.
 *  boundaries : [n_inner = n_levels - 1]  float32 inner Lloyd-Max boundaries.
 *  codes      : same shape as x_rot, uint8 in [0, n_levels - 1].
 *  sigma      : x_rot.shape with last axis = 1, float32 (per-row scale
 *               σ = ||x|| / √D).
 *
 * Returns 0 on success.
 */
int mlx_lumen_turboquant_encode_fused(
    mlx_array* codes,
    mlx_array* sigma,
    mlx_array x_rot,
    mlx_array boundaries,
    mlx_stream s);

/**
 * TurboQuant Stage-1 fused with rotation: (x @ R) → σ → normalize →
 * Lloyd-Max encode, all in one kernel. Equivalent to
 * (rotate_last_axis; encode_fused) but skips the bf16 intermediate plus the
 * separate matmul/cast dispatches.
 *
 *  x_bf16     : [..., D]  bfloat16, un-rotated input. D ∈ multiples of 32,
 *               ≤ 1024.
 *  R_f32      : [D, D]    float32, Haar orthogonal rotation matrix.
 *  boundaries : [n_inner = n_levels - 1]  float32 inner Lloyd-Max
 *               boundaries.
 *  codes      : same shape as x_bf16, uint8 in [0, n_levels - 1].
 *  sigma      : x_bf16.shape with last axis = 1, float32 (per-row scale
 *               σ = ||x_rot|| / √D).
 *
 * Returns 0 on success.
 */
int mlx_lumen_turboquant_rot_encode_fused(
    mlx_array* codes,
    mlx_array* sigma,
    mlx_array x_bf16,
    mlx_array R_f32,
    mlx_array boundaries,
    mlx_stream s);

/**
 * TurboQuant Q @ K_codes inline matmul: computes attention scores
 * S[B,H,T,N] = Q · K_dq^T without ever materializing K_dq. Inline Lloyd-Max
 * dequant via centroids LUT + per-K-vector σ.
 *
 *  q         : [B, H,    T, D=256]   bfloat16 queries.
 *  k_codes   : [B, H_kv, N, D=256]   uint8 Lloyd-Max codes (bits ≤ 4).
 *  k_sigma   : [B, H_kv, N]          float32 per-K-vector σ. The cache
 *               typically stores sigma with a trailing-1 axis; reshape /
 *               squeeze before calling.
 *  centroids : [n_levels = 2^bits]   float32 Lloyd-Max codebook.
 *  scores    : [B, H,    T, N]       bfloat16 (Q·K_dq^T).
 *
 * Constraints: D == 256, n_levels ≤ 16, H % H_kv == 0.
 * Returns 0 on success.
 */
int mlx_lumen_turboquant_qk_inline(
    mlx_array* scores,
    mlx_array q,
    mlx_array k_codes,
    mlx_array k_sigma,
    mlx_array centroids,
    mlx_stream s);

/**
 * TurboQuant softmax_scores @ V_codes inline matmul: computes attention
 * output O[B,H,T,D] = S · V_dq without materializing V_dq. Symmetric V-side
 * counterpart to mlx_lumen_turboquant_qk_inline. Inline Lloyd-Max dequant
 * via centroids LUT + per-V-vector σ.
 *
 *  s         : [B, H,    T, N]       bfloat16 (softmax-normalized scores).
 *  v_codes   : [B, H_kv, N, D=256]   uint8 Lloyd-Max codes (bits ≤ 4).
 *  v_sigma   : [B, H_kv, N]          float32 per-V-vector σ.
 *  centroids : [n_levels = 2^bits]   float32 Lloyd-Max codebook.
 *  out       : [B, H,    T, D]       bfloat16 (attention output).
 *
 * Constraints: D == 256, n_levels ≤ 16, H % H_kv == 0.
 * Returns 0 on success.
 */
/**
 * TurboQuant Stage-1 fused encode with PACKED 4-bit output. Same as
 * mlx_lumen_turboquant_encode_fused but emits codes as uint32 with 8
 * codes per word (4-bit each, low-nibble-first).
 *
 *  x_rot      : [..., D]    bfloat16 (D ∈ multiples of 32 and 8, ≤ 1024)
 *  boundaries : [15]        float32 (must be 4-bit boundaries → n_levels=16)
 *  codes_pkd  : [..., D/8]  uint32 (packed)
 *  sigma      : [..., 1]    float32
 *
 * Returns 0 on success.
 */
int mlx_lumen_turboquant_encode_fused_packed4(
    mlx_array* codes_pkd,
    mlx_array* sigma,
    mlx_array x_rot,
    mlx_array boundaries,
    mlx_stream stream);

/**
 * TurboQuant Q @ K_codes_packed inline matmul (4-bit packed). Same shape
 * contract as mlx_lumen_turboquant_qk_inline but K_codes is packed uint32
 * (8 codes per word). Halves K_codes DRAM read.
 *
 *  q           : [B, H,    T, D=256]      bfloat16
 *  k_codes_pkd : [B, H_kv, N, D/8 = 32]   uint32 (PACKED 4-bit)
 *  k_sigma     : [B, H_kv, N]             float32
 *  centroids   : [16]                     float32
 *  scores      : [B, H,    T, N]          bfloat16
 *
 * Returns 0 on success.
 */
int mlx_lumen_turboquant_qk_inline_packed4(
    mlx_array* scores,
    mlx_array q,
    mlx_array k_codes_pkd,
    mlx_array k_sigma,
    mlx_array centroids,
    mlx_stream stream);

int mlx_lumen_turboquant_sv_inline(
    mlx_array* out,
    mlx_array s,
    mlx_array v_codes,
    mlx_array v_sigma,
    mlx_array centroids,
    mlx_stream stream);

/**
 * Fused TurboQuant attention: O = softmax(Q * K_dq^T * scale) * V_dq
 * in a single Metal dispatch with inline Lloyd-Max K/V dequant.
 *
 *  q         : [B, H,    T=1, D]   bfloat16
 *  k_codes   : [B, H_kv, N,   D]   uint8
 *  k_sigma   : [B, H_kv, N]        float32
 *  v_codes   : [B, H_kv, N,   D]   uint8
 *  v_sigma   : [B, H_kv, N]        float32
 *  centroids : [n_levels <= 16]    float32
 *  scale     : f32 host scalar (attention scale)
 *  out       : [B, H,    T=1, D]   bfloat16
 *
 * Constraints: T=1, D in {256, 512}, n_levels <= 16, H % H_kv == 0.
 * Returns 0 on success.
 */
int mlx_lumen_turboquant_fused_attn(
    mlx_array* out,
    mlx_array q,
    mlx_array k_codes,
    mlx_array k_sigma,
    mlx_array v_codes,
    mlx_array v_sigma,
    mlx_array centroids,
    float scale,
    mlx_stream stream);

#ifdef __cplusplus
}
#endif

#endif /* MLX_LUMEN_H */
