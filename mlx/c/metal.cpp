/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/metal.h"
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/enums.h"
#include "mlx/c/private/mlx.h"
#include "mlx/device.h"
#include "mlx/stream.h"

extern "C" int mlx_metal_is_available(bool* res) {
  try {
    *res = mlx::core::metal::is_available();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_start_capture(const char* path) {
  try {
    mlx::core::metal::start_capture(std::string(path));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_stop_capture(void) {
  try {
    mlx::core::metal::stop_capture();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// lumen-rs Phase 1.5 deep-dive — kernel cache stats C wrappers.
extern "C" int mlx_metal_kernel_cache_hits(uint64_t* res) {
  try {
    *res = mlx::core::metal::kernel_cache_hits();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_kernel_cache_misses(uint64_t* res) {
  try {
    *res = mlx::core::metal::kernel_cache_misses();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_reset_kernel_cache_stats(void) {
  try {
    mlx::core::metal::reset_kernel_cache_stats();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// lumen-rs Phase 1.5 Step C: command buffer batching C wrappers.
extern "C" int mlx_metal_cmd_buffer_commits(uint64_t* res) {
  try {
    *res = mlx::core::metal::cmd_buffer_commits();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_cmd_buffer_ops_total(uint64_t* res) {
  try {
    *res = mlx::core::metal::cmd_buffer_ops_total();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_reset_cmd_buffer_stats(void) {
  try {
    mlx::core::metal::reset_cmd_buffer_stats();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// lumen-rs Phase 1.5 Step D (H2): scheduler contention C wrappers.
extern "C" int mlx_metal_scheduler_new_task_count(uint64_t* res) {
  try {
    *res = mlx::core::metal::scheduler_new_task_count();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_scheduler_completion_count(uint64_t* res) {
  try {
    *res = mlx::core::metal::scheduler_completion_count();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_scheduler_lock_wait_ns(uint64_t* res) {
  try {
    *res = mlx::core::metal::scheduler_lock_wait_ns();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_scheduler_max_active_tasks(int32_t* res) {
  try {
    *res = mlx::core::metal::scheduler_max_active_tasks();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_reset_scheduler_stats(void) {
  try {
    mlx::core::metal::reset_scheduler_stats();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// Step E wrappers.
extern "C" int mlx_metal_eval_gpu_calls(uint64_t* res) {
  try {
    *res = mlx::core::metal::eval_gpu_calls();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_eval_gpu_ns(uint64_t* res) {
  try {
    *res = mlx::core::metal::eval_gpu_ns();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_reset_eval_gpu_stats(void) {
  try {
    mlx::core::metal::reset_eval_gpu_stats();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// Step F wrappers.
#define PRIM_HIST_WRAPPER(NAME) \
  extern "C" int mlx_metal_prim_hist_##NAME(uint64_t* res) { \
    try { *res = mlx::core::metal::prim_hist_##NAME(); } \
    catch (std::exception& e) { mlx_error(e.what()); return 1; } \
    return 0; \
  }
PRIM_HIST_WRAPPER(rms_norm)
PRIM_HIST_WRAPPER(qmm)
PRIM_HIST_WRAPPER(reshape)
PRIM_HIST_WRAPPER(broadcast)
PRIM_HIST_WRAPPER(multiply)
PRIM_HIST_WRAPPER(transpose)
PRIM_HIST_WRAPPER(compiled)
PRIM_HIST_WRAPPER(other)
#undef PRIM_HIST_WRAPPER

extern "C" int mlx_metal_reset_prim_hist(void) {
  try {
    mlx::core::metal::reset_prim_hist();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_prim_hist_dump_dynamic(
    char* buf, int buf_size, int* written) {
  try {
    int n = mlx::core::metal::prim_hist_dump_dynamic(buf, buf_size);
    if (n < 0) {
      *written = 0;
      return 2; // truncation
    }
    *written = n;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_metal_reset_prim_hist_dynamic(void) {
  try {
    mlx::core::metal::reset_prim_hist_dynamic();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// Step F3: AsType dtype-pair counters.
#define ASTYPE_PAIR_WRAPPER(NAME) \
  extern "C" int mlx_metal_astype_##NAME(uint64_t* res) { \
    try { *res = mlx::core::metal::astype_##NAME(); } \
    catch (std::exception& e) { mlx_error(e.what()); return 1; } \
    return 0; \
  }
ASTYPE_PAIR_WRAPPER(bf16_to_f32)
ASTYPE_PAIR_WRAPPER(f32_to_bf16)
ASTYPE_PAIR_WRAPPER(noop)
ASTYPE_PAIR_WRAPPER(other_pair)
#undef ASTYPE_PAIR_WRAPPER

extern "C" int mlx_metal_reset_astype_pair(void) {
  try {
    mlx::core::metal::reset_astype_pair();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// lumen-rs Phase 1.8 M1.5 — Metal buffer alloc + array wrapping bridge.

extern "C" void* _mlx_metal_allocator_malloc(size_t size_bytes) {
  try {
    auto buf = mlx::core::allocator::malloc(size_bytes);
    return buf.ptr();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}

extern "C" void _mlx_metal_allocator_free(void* mtl_buffer_ptr) {
  try {
    if (mtl_buffer_ptr == nullptr) {
      return;
    }
    mlx::core::allocator::Buffer buf(mtl_buffer_ptr);
    mlx::core::allocator::free(buf);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
}

extern "C" void* _mlx_metal_current_command_buffer() {
  try {
    auto stream = mlx::core::default_stream(
        mlx::core::Device(mlx::core::Device::gpu));
    auto& device = mlx::core::metal::device(
        mlx::core::Device(mlx::core::Device::gpu));
    return static_cast<void*>(device.get_command_buffer(stream.index));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}

extern "C" void* _mlx_metal_default_command_queue() {
  try {
    auto stream = mlx::core::default_stream(
        mlx::core::Device(mlx::core::Device::gpu));
    auto& device = mlx::core::metal::device(
        mlx::core::Device(mlx::core::Device::gpu));
    return static_cast<void*>(device.get_queue(stream));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}

extern "C" mlx_array _mlx_array_new_from_metal_buffer(
    void* mtl_buffer_ptr,
    const int* shape,
    int dim,
    mlx_dtype dtype) {
  try {
    if (mtl_buffer_ptr == nullptr) {
      mlx_error("_mlx_array_new_from_metal_buffer: null buffer");
      return mlx_array_new_();
    }
    mlx::core::Shape cpp_shape(shape, shape + dim);
    mlx::core::Dtype cpp_dtype = mlx_dtype_to_cpp(dtype);
    mlx::core::allocator::Buffer buf(mtl_buffer_ptr);
    return mlx_array_new_(mlx::core::array(
        buf, std::move(cpp_shape), cpp_dtype, mlx::core::allocator::free));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_new_();
  }
}
