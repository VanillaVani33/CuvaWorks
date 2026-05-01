#pragma once


// handles random things
// reimplements uses Cuda's CuRand Kernel on the CPU if CUDA is not defined
// this is done so that random calls can be consistent between the cpu and gpu 
#include "cuvaworks_types.h"
#ifdef CUVAWORKS_CUDA
#include "curand_kernel.h"


typedef curandState CuvaRand;

_cuvadev void initCuvarand(unsigned long long seed,
    unsigned long long subsequence,
    unsigned long long offset,
    CuvaRand* state) {

    curand_init(seed, subsequence, offset, state);
}

_cuvadev float uniCuvarand(CuvaRand* cuva) {
    return curand_uniform(cuva);
}

_cuvadev float uniCuvarand(CuvaRand* cuva, float a, float b) {
    return curand_uniform(cuva) * (b - a) + a;
}


#else
#include "cuvarand_precalcs.h"
struct CuvaRand {
    unsigned int d, v[5];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};


template<int N>
_cuvadev void __cuvarand_matvec_inplace(unsigned int* vector, unsigned int* matrix) {
    unsigned int result[N] = { 0 };
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 32; j++) {
            if (vector[i] & (1 << j)) {
                for (int k = 0; k < N; k++) {
                    result[k] ^= matrix[N * (i * 32 + j) + k];
                }
            }
        }
    }
    for (int i = 0; i < N; i++) {
        vector[i] = result[i];
    }
}

template <typename T, int N>
_cuvadev void cuvarand_skipahead_sequence_inplace(unsigned long long x, T* state) {
    int matrix_num = 0;
    while (x) {
        for (unsigned int t = 0; t < (x & PRECALC_BLOCK_MASK); t++) {
            __cuvarand_matvec_inplace<N>(state->v, precalc_xorwow_matrix_host[matrix_num]);
        }
        x >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    /* No update of state->d needed, guaranteed to be a multiple of 2^32 */
}

template <typename T, int N>
_cuvadev void cuvarand_skipahead_inplace(const unsigned long long x, T* state)
{
    unsigned long long p = x;
    int matrix_num = 0;
    while (p) {
        for (unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
            __cuvarand_matvec_inplace<N>(state->v, precalc_xorwow_offset_matrix_host[matrix_num]);
        }
        p >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    state->d += 362437 * (unsigned int)x;
}

// initialize cuvarand
_cuvadev void initCuvarand(unsigned long long seed,
    unsigned long long subsequence,
    unsigned long long offset,
    CuvaRand* state) {

    // Break up seed, apply salt
    // Constants are arbitrary nonzero values
    unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
    unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
    // Simple multiplication to mix up bits
    // Constants are arbitrary odd values
    unsigned int t0 = 1099087573UL * s0;
    unsigned int t1 = 2591861531UL * s1;
    state->d = 6615241 + t1 + t0;
    state->v[0] = 123456789UL + t0;
    state->v[1] = 362436069UL ^ t0;
    state->v[2] = 521288629UL + t1;
    state->v[3] = 88675123UL ^ t1;
    state->v[4] = 5783321UL + t0;
    cuvarand_skipahead_sequence_inplace<CuvaRand, 5>(subsequence, state);
    cuvarand_skipahead_inplace<CuvaRand, 5>(offset, state);
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra = 0.f;
    state->boxmuller_extra_double = 0.;
}

/**
 * \brief Return 32-bits of pseudorandomness from an XORWOW generator.
 *
 * Return 32-bits of pseudorandomness from the XORWOW generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
 */
_cuvadev unsigned int cuvarand(CuvaRand* state)
{
    unsigned int t;
    t = (state->v[0] ^ (state->v[0] >> 2));
    state->v[0] = state->v[1];
    state->v[1] = state->v[2];
    state->v[2] = state->v[3];
    state->v[3] = state->v[4];
    state->v[4] = (state->v[4] ^ (state->v[4] << 4)) ^ (t ^ (t << 1));
    state->d += 362437;
    return state->v[4] + state->d;
}


_cuvadev float _uniCuvarand(unsigned int x) {
    // constant from curand
    return x * (2.3283064e-10f) + ((2.3283064e-10f) / 2.0f);
}

/**
 * \brief Return a uniformly distributed float from an XORWOW generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p curand() to
 * get enough random bits to create the return value.  The current
 * implementation uses one call.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */

 // returns a random number a <= x <= b
_cuvadev float uniCuvarand(CuvaRand* cuva, float a, float b) {
    return _uniCuvarand(cuvarand(cuva)) * (b - a) + a;
}

_cuvadev float uniCuvarand(CuvaRand* cuva) {
    return _uniCuvarand(cuvarand(cuva));
}

#endif

