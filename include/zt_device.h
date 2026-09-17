/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-002
   File:              zt_device.h
   Parent-Work:       zero-token
   Copyright:         2026 BEL ESPRIT D ACCORD
   License-ID:        GPL-2.0 OR AGPL-3.0
   Covenant-Version:  1.0
   Compliance:        FAIL-CLOSED

   This file is governed by the GNU General Public License version 2.0 or the GNU Affero
   General Public License version 3.0, at your option, together with the applicable Sovereign
   Leviathan Covenant.

   Whatsoever branch this root shall bear,
   Must breathe the exact and sovereign air.
   Touch but a leaf, invoke a single thread,
   And honor still the terms beneath it spread.

   Ignorantia juris non excusat.

   Clone-Gate: sha256:3568a8fbb3c16adccc07d2506ff05051c9d059adcff506d3611d0219bcdd5d24

   See: LICENSE

   -------------------------------------------------------------------------
   zt_device.h — target abstraction layer.

   Exactly one of three compilers includes this header:
     - a C99 host compiler          (the C sources, tests, bench)
     - nvcc                          (kernels/zt_kernels.cuh via src/zt_cuda.cu)
     - the Metal shading compiler    (metal/zt_kernels.metal)

   It defines the small macro vocabulary that lets the shared decision core
   (include/zt_core.h) compile *unchanged* on all three.  Nothing else in the
   project may test the compiler directly; it must go through these macros.

     ZT_FN            function qualifier (static inline / __host__ __device__)
     ZT_DEV           address-space qualifier for global-memory pointers
     ZT_CONST         address-space qualifier for constant-memory pointers
     ZT_THREAD        address-space qualifier for per-thread (stack) pointers
     ZT_EXPF/ZT_LOGF  IEEE-precise exp/log (no fast-math: determinism matters)
     ZT_FMAXF/FMINF   float min/max
     ZT_INF           +infinity as float
     ZT_BITS_TO_F32   reinterpret uint32 bits as float
     ZT_F16_TO_F32    decode IEEE binary16 bits to float
   ======================================================================== */
#ifndef ZT_DEVICE_H
#define ZT_DEVICE_H

#if defined(__METAL_VERSION__)
/* ---------------------------------------------------------------- Metal */
#  include <metal_stdlib>
#  define ZT_TARGET_METAL 1
#  define ZT_FN             static inline
#  define ZT_DEV            device
#  define ZT_CONST          constant
#  define ZT_THREAD         thread
#  define ZT_EXPF(x)        metal::precise::exp(x)
#  define ZT_LOGF(x)        metal::precise::log(x)
#  define ZT_FMAXF(a, b)    metal::fmax((a), (b))
#  define ZT_FMINF(a, b)    metal::fmin((a), (b))
#  define ZT_INF            INFINITY
#  define ZT_BITS_TO_F32(u) metal::as_type<float>((uint32_t)(u))
#  define ZT_F16_TO_F32(h)  ((float)metal::as_type<half>((uint16_t)(h)))

#elif defined(__CUDACC__)
/* ---------------------------------------------------------------- CUDA */
#  include <stdint.h>
#  include <string.h>
#  include <math.h>
#  include <cuda_fp16.h>
#  define ZT_TARGET_CUDA 1
#  define ZT_FN             static __host__ __device__ __forceinline__
#  define ZT_DEV
#  define ZT_CONST
#  define ZT_THREAD
#  define ZT_EXPF(x)        expf(x)
#  define ZT_LOGF(x)        logf(x)
#  define ZT_FMAXF(a, b)    fmaxf((a), (b))
#  define ZT_FMINF(a, b)    fminf((a), (b))
#  define ZT_INF            INFINITY
ZT_FN float zt_bits_to_f32_(uint32_t u)
{
#  if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#  else
    float f; memcpy(&f, &u, sizeof f); return f;
#  endif
}
ZT_FN uint32_t zt_f32_to_bits_(float f)
{
#  if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#  else
    uint32_t u; memcpy(&u, &f, sizeof u); return u;
#  endif
}
ZT_FN float zt_f16_to_f32_(uint16_t h)
{
    __half_raw r; r.x = h;
    return __half2float(__half(r));
}
#  define ZT_BITS_TO_F32(u) zt_bits_to_f32_((uint32_t)(u))
#  define ZT_F16_TO_F32(h)  zt_f16_to_f32_((uint16_t)(h))

#else
/* ---------------------------------------------------------------- host C99 */
#  include <stdint.h>
#  include <string.h>
#  include <math.h>
#  define ZT_TARGET_HOST 1
#  define ZT_FN             static inline
#  define ZT_DEV
#  define ZT_CONST
#  define ZT_THREAD
#  define ZT_EXPF(x)        expf(x)
#  define ZT_LOGF(x)        logf(x)
#  define ZT_FMAXF(a, b)    fmaxf((a), (b))
#  define ZT_FMINF(a, b)    fminf((a), (b))
#  define ZT_INF            INFINITY
ZT_FN float    zt_bits_to_f32_(uint32_t u) { float f; memcpy(&f, &u, sizeof f); return f; }
ZT_FN uint32_t zt_f32_to_bits_(float f)    { uint32_t u; memcpy(&u, &f, sizeof u); return u; }
/* IEEE binary16 -> binary32, exact (handles subnormals, inf, nan). */
ZT_FN float zt_f16_to_f32_(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t man  = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;
        } else {                                  /* subnormal: renormalise */
            exp = 113u;                           /* 127 - 15 + 1 */
            while ((man & 0x400u) == 0u) { man <<= 1; exp--; }
            man &= 0x3ffu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (man << 13);  /* inf / nan */
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return zt_bits_to_f32_(bits);
}
#  define ZT_BITS_TO_F32(u) zt_bits_to_f32_((uint32_t)(u))
#  define ZT_F16_TO_F32(h)  zt_f16_to_f32_((uint16_t)(h))
#endif

#if !defined(ZT_TARGET_METAL)
/* binary32 -> binary16 bits, round-to-nearest-even.  Host and CUDA-host only;
 * Metal has native half.  Used to build the fp16 head slice for tensor-core paths. */
ZT_FN uint16_t zt_f32_to_f16_(float f)
{
    uint32_t x    = zt_f32_to_bits_(f);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp  = (x >> 23) & 0xffu;
    uint32_t man  = x & 0x7fffffu;
    int32_t  e;
    if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
    e = (int32_t)exp - 127 + 15;
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        uint32_t shift, hm, rem, halfway;
        if (e < -10) return (uint16_t)sign;
        man |= 0x800000u;
        shift   = (uint32_t)(14 - e);
        hm      = man >> shift;
        rem     = man & ((1u << shift) - 1u);
        halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (hm & 1u))) hm++;
        return (uint16_t)(sign | hm);
    }
    {
        uint32_t hm  = man >> 13;
        uint32_t rem = man & 0x1fffu;
        uint16_t out = (uint16_t)(sign | ((uint32_t)e << 10) | hm);
        if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) out = (uint16_t)(out + 1u);
        return out;
    }
}
#endif

#endif /* ZT_DEVICE_H */
