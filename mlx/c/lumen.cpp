// lumen-rs Phase 1.8 M4.8 — C ABI impl for mlx_lumen_flash_attn_bf16.

#include "mlx/c/lumen.h"

#include <optional>

#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/fast.h"
#include "mlx/lumen_primitives.h"

extern "C" int mlx_lumen_flash_attn_bf16(
    mlx_array* result,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    int has_mask,
    mlx_array mask,
    float scale,
    mlx_stream s) {
  try {
    const auto& q_ref = mlx_array_get_(q);
    const auto& k_ref = mlx_array_get_(k);
    const auto& v_ref = mlx_array_get_(v);

    std::optional<mlx::core::array> mask_opt = std::nullopt;
    if (has_mask) {
      if (mask.ctx == nullptr) {
        mlx_error(
            "mlx_lumen_flash_attn_bf16: has_mask=1 but mask handle is null");
        return 1;
      }
      mask_opt = mlx_array_get_(mask);
    }

    auto stream_ref = mlx_stream_get_(s);
    auto out = mlx::core::lumen::flash_attn_bf16(
        q_ref, k_ref, v_ref, scale, std::move(mask_opt), stream_ref);
    mlx_array_set_(*result, out);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_flash_attn_prefill_bf16(
    mlx_array* result,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    float scale,
    uint32_t window_size,
    uint32_t kv_offset,
    mlx_stream s) {
  try {
    const auto& q_ref = mlx_array_get_(q);
    const auto& k_ref = mlx_array_get_(k);
    const auto& v_ref = mlx_array_get_(v);
    auto stream_ref = mlx_stream_get_(s);

    auto out = mlx::core::lumen::flash_attn_prefill_bf16(
        q_ref, k_ref, v_ref, scale, window_size, kv_offset, stream_ref);
    mlx_array_set_(*result, out);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// Windowed scaled-dot-product attention: thin wrapper around
// mlx::core::fast::scaled_dot_product_attention with mask_mode="causal" and
// window_size>0. The steel kernel's kb_start truncation + left-edge mask
// activate via the function constant has_window=true when window_size > 0,
// skipping all-masked-out K blocks below the band's lower edge.
extern "C" int mlx_lumen_turboquant_encode(
    mlx_array* result,
    mlx_array x_norm,
    mlx_array boundaries,
    mlx_stream s) {
  try {
    const auto& x_ref = mlx_array_get_(x_norm);
    const auto& bnd_ref = mlx_array_get_(boundaries);
    auto stream_ref = mlx_stream_get_(s);
    auto out =
        mlx::core::lumen::turboquant_encode(x_ref, bnd_ref, stream_ref);
    mlx_array_set_(*result, out);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_qjl_pack_signs(
    mlx_array* result,
    mlx_array values,
    int m,
    mlx_stream s) {
  try {
    const auto& v_ref = mlx_array_get_(values);
    auto stream_ref = mlx_stream_get_(s);
    auto out = mlx::core::lumen::qjl_pack_signs(v_ref, m, stream_ref);
    mlx_array_set_(*result, out);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_qjl_unpack_signs(
    mlx_array* result,
    mlx_array packed,
    int m,
    mlx_stream s) {
  try {
    const auto& p_ref = mlx_array_get_(packed);
    auto stream_ref = mlx_stream_get_(s);
    auto out = mlx::core::lumen::qjl_unpack_signs(p_ref, m, stream_ref);
    mlx_array_set_(*result, out);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_turboquant_encode_fused(
    mlx_array* codes,
    mlx_array* sigma,
    mlx_array x_rot,
    mlx_array boundaries,
    mlx_stream s) {
  try {
    const auto& x_ref = mlx_array_get_(x_rot);
    const auto& bnd_ref = mlx_array_get_(boundaries);
    auto stream_ref = mlx_stream_get_(s);
    auto outs = mlx::core::lumen::turboquant_encode_fused(
        x_ref, bnd_ref, stream_ref);
    if (outs.size() != 2) {
      mlx_error(
          "mlx_lumen_turboquant_encode_fused: expected 2 outputs from "
          "primitive");
      return 1;
    }
    mlx_array_set_(*codes, outs[0]);
    mlx_array_set_(*sigma, outs[1]);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_turboquant_rot_encode_fused(
    mlx_array* codes,
    mlx_array* sigma,
    mlx_array x_bf16,
    mlx_array R_f32,
    mlx_array boundaries,
    mlx_stream s) {
  try {
    const auto& x_ref = mlx_array_get_(x_bf16);
    const auto& r_ref = mlx_array_get_(R_f32);
    const auto& bnd_ref = mlx_array_get_(boundaries);
    auto stream_ref = mlx_stream_get_(s);
    auto outs = mlx::core::lumen::turboquant_rot_encode_fused(
        x_ref, r_ref, bnd_ref, stream_ref);
    if (outs.size() != 2) {
      mlx_error(
          "mlx_lumen_turboquant_rot_encode_fused: expected 2 outputs from "
          "primitive");
      return 1;
    }
    mlx_array_set_(*codes, outs[0]);
    mlx_array_set_(*sigma, outs[1]);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_turboquant_qk_inline(
    mlx_array* scores,
    mlx_array q,
    mlx_array k_codes,
    mlx_array k_sigma,
    mlx_array centroids,
    mlx_stream s) {
  try {
    const auto& q_ref         = mlx_array_get_(q);
    const auto& kc_ref        = mlx_array_get_(k_codes);
    const auto& ks_ref        = mlx_array_get_(k_sigma);
    const auto& centroids_ref = mlx_array_get_(centroids);
    auto stream_ref = mlx_stream_get_(s);
    auto out = mlx::core::lumen::turboquant_qk_inline(
        q_ref, kc_ref, ks_ref, centroids_ref, stream_ref);
    mlx_array_set_(*scores, out);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_turboquant_encode_fused_packed4(
    mlx_array* codes_pkd,
    mlx_array* sigma,
    mlx_array x_rot,
    mlx_array boundaries,
    mlx_stream stream) {
  try {
    const auto& x_ref   = mlx_array_get_(x_rot);
    const auto& bnd_ref = mlx_array_get_(boundaries);
    auto stream_ref = mlx_stream_get_(stream);
    auto outs = mlx::core::lumen::turboquant_encode_fused_packed4(
        x_ref, bnd_ref, stream_ref);
    if (outs.size() != 2) {
      mlx_error(
          "mlx_lumen_turboquant_encode_fused_packed4: expected 2 outputs");
      return 1;
    }
    mlx_array_set_(*codes_pkd, outs[0]);
    mlx_array_set_(*sigma, outs[1]);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_turboquant_qk_inline_packed4(
    mlx_array* scores,
    mlx_array q,
    mlx_array k_codes_pkd,
    mlx_array k_sigma,
    mlx_array centroids,
    mlx_stream stream) {
  try {
    const auto& q_ref         = mlx_array_get_(q);
    const auto& kc_ref        = mlx_array_get_(k_codes_pkd);
    const auto& ks_ref        = mlx_array_get_(k_sigma);
    const auto& centroids_ref = mlx_array_get_(centroids);
    auto stream_ref = mlx_stream_get_(stream);
    auto out = mlx::core::lumen::turboquant_qk_inline_packed4(
        q_ref, kc_ref, ks_ref, centroids_ref, stream_ref);
    mlx_array_set_(*scores, out);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_turboquant_sv_inline(
    mlx_array* out,
    mlx_array s,
    mlx_array v_codes,
    mlx_array v_sigma,
    mlx_array centroids,
    mlx_stream stream) {
  try {
    const auto& s_ref         = mlx_array_get_(s);
    const auto& vc_ref        = mlx_array_get_(v_codes);
    const auto& vs_ref        = mlx_array_get_(v_sigma);
    const auto& centroids_ref = mlx_array_get_(centroids);
    auto stream_ref = mlx_stream_get_(stream);
    auto result = mlx::core::lumen::turboquant_sv_inline(
        s_ref, vc_ref, vs_ref, centroids_ref, stream_ref);
    mlx_array_set_(*out, result);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_lumen_sdpa_windowed(
    mlx_array* result,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    float scale,
    int window_size,
    mlx_stream s) {
  try {
    const auto& q_ref = mlx_array_get_(q);
    const auto& k_ref = mlx_array_get_(k);
    const auto& v_ref = mlx_array_get_(v);
    auto stream_ref = mlx_stream_get_(s);

    auto out = mlx::core::fast::scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        scale,
        /* mask_mode = */ "causal",
        /* mask_arr  = */ std::nullopt,
        /* sinks     = */ std::nullopt,
        /* window_size = */ window_size,
        stream_ref);
    mlx_array_set_(*result, out);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
