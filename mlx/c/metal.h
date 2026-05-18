/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_METAL_H
#define MLX_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup metal Metal specific operations
 */
/**@{*/

int mlx_metal_is_available(bool* res);
int mlx_metal_start_capture(const char* path);
int mlx_metal_stop_capture(void);
/* lumen-rs Phase 1.5 deep-dive — kernel cache stats. */
int mlx_metal_kernel_cache_hits(uint64_t* res);
int mlx_metal_kernel_cache_misses(uint64_t* res);
int mlx_metal_reset_kernel_cache_stats(void);
/* Step C: command buffer batching stats. */
int mlx_metal_cmd_buffer_commits(uint64_t* res);
int mlx_metal_cmd_buffer_ops_total(uint64_t* res);
int mlx_metal_reset_cmd_buffer_stats(void);
/* Step D (H2): scheduler contention stats. */
int mlx_metal_scheduler_new_task_count(uint64_t* res);
int mlx_metal_scheduler_completion_count(uint64_t* res);
int mlx_metal_scheduler_lock_wait_ns(uint64_t* res);
int mlx_metal_scheduler_max_active_tasks(int32_t* res);
int mlx_metal_reset_scheduler_stats(void);
/* Step E: per-primitive gpu::eval encode time. */
int mlx_metal_eval_gpu_calls(uint64_t* res);
int mlx_metal_eval_gpu_ns(uint64_t* res);
int mlx_metal_reset_eval_gpu_stats(void);
/* Step F: primitive-type histogram. */
int mlx_metal_prim_hist_rms_norm(uint64_t* res);
int mlx_metal_prim_hist_qmm(uint64_t* res);
int mlx_metal_prim_hist_reshape(uint64_t* res);
int mlx_metal_prim_hist_broadcast(uint64_t* res);
int mlx_metal_prim_hist_multiply(uint64_t* res);
int mlx_metal_prim_hist_transpose(uint64_t* res);
int mlx_metal_prim_hist_compiled(uint64_t* res);
int mlx_metal_prim_hist_other(uint64_t* res);
int mlx_metal_reset_prim_hist(void);
/* Step F2: dynamic histogram. */
int mlx_metal_prim_hist_dump_dynamic(char* buf, int buf_size, int* written);
int mlx_metal_reset_prim_hist_dynamic(void);
/* Step F3: AsType dtype-pair counters. */
int mlx_metal_astype_bf16_to_f32(uint64_t* res);
int mlx_metal_astype_f32_to_bf16(uint64_t* res);
int mlx_metal_astype_noop(uint64_t* res);
int mlx_metal_astype_other_pair(uint64_t* res);
int mlx_metal_reset_astype_pair(void);

/* lumen-rs Phase 1.8 M1.5 — Metal buffer alloc + array wrapping bridge.
 *
 * Lets external Metal kernels (turboquant-metal) allocate buffers via mlx's
 * global MetalAllocator so the resulting buffers participate in mlx's
 * residency tracking, then wrap them as mlx_array without copying.
 */

/* Allocate a Metal buffer via mlx's global MetalAllocator.
 * Returns the internal MTL::Buffer* cast as void* (same convention as
 * _mlx_array_metal_buffer above). The buffer is registered in mlx's
 * residency set and active_memory counters.
 * Returns nullptr on failure.
 *
 * Ownership: caller owns the buffer. Either pass it to
 * _mlx_array_new_from_metal_buffer (array adopts ownership) or free it
 * with _mlx_metal_allocator_free().
 */
void* _mlx_metal_allocator_malloc(size_t size_bytes);

/* Free a Metal buffer previously obtained from _mlx_metal_allocator_malloc.
 * Do NOT call if ownership was transferred to an mlx_array via
 * _mlx_array_new_from_metal_buffer — the array's destructor will free it.
 */
void _mlx_metal_allocator_free(void* mtl_buffer_ptr);

/* Returns the `MTL::CommandBuffer*` mlx is currently encoding into for the
 * default GPU stream, cast as `void*`. Lazy-creates one if the stream's
 * buffer slot is empty. Custom Metal kernels that encode into this CB share
 * mlx's batching: a single commit (by mlx's scheduler) drains both our
 * dispatches and mlx's, with implicit serialization in submission order
 * across encoders on the same CB.
 *
 * Returns nullptr if Metal is unavailable or CB allocation fails.
 *
 * # Caveats
 *
 * - DO NOT commit the returned CB. mlx owns its lifecycle.
 * - mlx may commit the returned CB at any time after this call (when its
 *   own op-count threshold trips). Encode + endEncoding promptly.
 * - The returned pointer is borrowed from mlx — do NOT retain/release
 *   outside the objc2 borrowed-reference pattern.
 */
void* _mlx_metal_current_command_buffer(void);

/* Returns the `MTL::CommandQueue*` mlx uses for the default GPU stream,
 * cast as `void*`. Custom Metal kernels that share this queue will be
 * serialized w.r.t. mlx's own dispatches without explicit fences:
 * command-buffer commits to the same queue execute in order.
 *
 * Returns nullptr if Metal is unavailable. Caller MUST NOT release or
 * retain the returned pointer outside the normal objc2 borrowed
 * reference pattern — mlx owns the queue's lifetime.
 */
void* _mlx_metal_default_command_queue(void);

/* Construct a new mlx_array that adopts (takes ownership of) a Metal
 * buffer obtained from _mlx_metal_allocator_malloc.
 *
 * size_bytes_check: pass `prod(shape) * dtype_size(dtype)` here — for
 * documentation; not strictly validated against the actual MTL::Buffer
 * length to allow strided/offset cases later.
 *
 * The buffer must have been allocated via mlx's MetalAllocator (so it is
 * in the residency set). When the returned mlx_array is destroyed, mlx's
 * default deleter (mlx::core::allocator::free) will release the buffer
 * correctly.
 */
mlx_array _mlx_array_new_from_metal_buffer(
    void* mtl_buffer_ptr,
    const int* shape,
    int dim,
    mlx_dtype dtype);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
