#pragma once
#include "cuvaworks_cuvakernel.h"
#include "malloc.h"
#include <string>


// MEMORY ALLOCATION INTERFACE ------------------------------------------------------------------

// allocates the given amount of memory
_cuvahos void** cuvaMalloc(size_t bytes);

// allocates the given amount of memory to a linked pointer
_cuvahos Linkp_void cuvaMallocLink(size_t bytes);

// copies data between source and destination
_cuvahos void cuvaCopy(void** dest, void** source, size_t bytes, MemcpyKind kind);

// copies data between the host and the device pointers
_cuvahos void cuvaCopy(Linkp_void p, size_t bytes, MemcpyKind kind);

// copies data from the host to the device pointer
_cuvahos void cuvaCopyToGPU(Linkp_void p, size_t bytes);

// copies data from the device to the host pointer
_cuvahos void cuvaCopyToCPU(Linkp_void p, size_t bytes);


// CUVA KERNEL INTERFACE -------------------------------------------------------------------------


// dispatches the given kernel asynchronously
_cuvahos void DispatchKernelAsync(Dim3 blocks, Dim3 threads, cuvakernel_t* kernel, void** args);

// dispatches the given kernel then synchronizes all threads
_cuvahos void DispatchKernel(Dim3 blocks, Dim3 threads, cuvakernel_t* kernel, void** args);


// gets cuva kernelID and stores it to "id"
_cuvadev void getKernelID(KernelID* id);

// waits for all threads being executed to complete
_cuvahos void cuvaSynchronize();

// frees the given device memory
_cuvahos void cuvaFree(void** dev_p);

_cuvahos void cuvaFree(Linkp_void lp);

// terminates the gpu device or the threadpool
_cuvahos void cuvaClose();

_cuvahos CuvaTarget getCuvaTarget();

_cuvahos const char* getCuvaTargetName();





// INTERFACE IMPLEMENTATION ---------------------------------------------------------------------------------------------------

// Implement fucntions for cuda device
#ifdef CUVAWORKS_CUDA

_cuvahos CuvaTarget getCuvaTarget() {
	return CuvaTarget::CuvaTarget_CudaDevice;
}
_cuvahos const char* getCuvaTargetName() {
	return "Cuda Device";
}
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
// 	checks for errors, then synchronizes. to be ran after calling a cuda kernel
#define CudaCheck() checkCudaErrors(cudaGetLastError());checkCudaErrors(cudaDeviceSynchronize())

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		printf("\nCUDA error = %i = %s at \n %s: %i\n%s\n\n", static_cast<unsigned int>(result), cudaGetErrorName(result), file, line, func);
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}



_cuvahos void** cuvaMalloc(size_t bytes) {
	void** temp;
	checkCudaErrors(cudaMalloc((void**)&temp, bytes));
	return temp;
}

_cuvahos Linkp_void cuvaMallocLink(size_t bytes) {
	return Linkp_void((void**)malloc(bytes), cuvaMalloc(bytes));
}

_cuvahos void** cuvaMallocHost(size_t bytes) {
	return (void**)malloc(bytes);
}
_cuvahos Linkp_void cuvaMallocHostLink(size_t bytes) {
	return Linkp_void(cuvaMallocHost(bytes), cuvaMallocHost(bytes));
}



_cuvahos void cuvaCopy(void** dest, void** source, size_t bytes, MemcpyKind kind) {

	cudaMemcpyKind ckind;

	switch (kind) {
	case MemcpyKind::MemcpyHostToHost:
		ckind = cudaMemcpyKind::cudaMemcpyHostToHost;
		break;
	case MemcpyKind::MemcpyHostToDevice:
		ckind = cudaMemcpyKind::cudaMemcpyHostToDevice;
		break;
	case MemcpyKind::MemcpyDeviceToHost:
		ckind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
		break;
	case MemcpyKind::MemcpyDeviceToDevice:
		ckind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
		break;
	case MemcpyKind::MemcpyDefault:
		ckind = cudaMemcpyKind::cudaMemcpyDefault;
		break;
	}

	checkCudaErrors(cudaMemcpy(dest, source, bytes, ckind));
}

_cuvahos void cuvaCopy(Linkp_void p, size_t bytes, MemcpyKind kind) {
	switch (kind) {
	case MemcpyKind::MemcpyHostToDevice:
		checkCudaErrors(cudaMemcpy(p.d, p.h, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
		break;
	case MemcpyKind::MemcpyDeviceToHost:
		checkCudaErrors(cudaMemcpy(p.h, p.d, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
		break;
	}
}



_cuvahos void cuvaCopyToGPU(Linkp_void p, size_t bytes) {
	checkCudaErrors(cudaMemcpy(p.d, p.h, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
}

_cuvahos void cuvaCopyToCPU(Linkp_void p, size_t bytes) {
	checkCudaErrors(cudaMemcpy(p.h, p.d, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
}


_cuvahos void cuvaSynchronize() {
	CudaCheck();
}

_cuvahos void cuvaClose() {
	cudaDeviceReset();
	exit(0);
}

_cuvahos void DispatchKernel(Dim3 blocks, Dim3 threads, cuvakernel_t* kernel, void** args) {
	kernel << <blocks, threads >> > (0, args);
	cuvaSynchronize();
}

_cuvahos void DispatchKernelAsync(Dim3 blocks, Dim3 threads, cuvakernel_t* kernel, void** args) {
	kernel << <blocks, threads >> > (0, args);
}

_cuvadev void getKernelID(KernelID* id) {
	id->i = threadIdx.x + blockIdx.x * blockDim.x;
	id->j = threadIdx.y + blockIdx.y * blockDim.y;
}

_cuvahos void cuvaFree(void** dev_p) {
	cudaFree(dev_p);
}

_cuvahos void cuvaFree(Linkp_void lp) {
	cuvaFree(lp.d);
	free(lp.h);
}


// cuda not defined, define functions for cpu
//#elif defined CUVAWORKS_CPU
#else

_cuvahos void** cuvaMalloc(size_t bytes) {
	return (void**)malloc(bytes);
}

_cuvahos void** cuvaMallocHost(size_t bytes) {
	return (void**)malloc(bytes);
}


_cuvahos Linkp_void cuvaMallocHostLink(size_t bytes) {
	return Linkp_void(cuvaMallocHost(bytes), cuvaMallocHost(bytes));
}

_cuvahos Linkp_void cuvaMallocLink(size_t bytes) {
	void** temp = (void**)malloc(bytes);
	return Linkp_void(temp, temp);
}


_cuvahos void cuvaCopy(void** dest, void** source, size_t bytes, MemcpyKind kind) {
	memcpy(dest, source, bytes);
}

_cuvahos void cuvaCopy(Linkp_void p, size_t bytes, MemcpyKind kind) {
	// no need to copy if linked pointer points to the same address
	// this is checked since on the cpu, linked pointers point to the 
	// same address by default.
	if (size_t(p.h) == size_t(p.d)) return;

	switch (kind) {
	case MemcpyKind::MemcpyHostToDevice:
		memcpy(p.d, p.h, bytes);
		break;
	case MemcpyKind::MemcpyDeviceToHost:
		memcpy(p.h, p.d, bytes);
		break;
	case MemcpyKind::MemcpyHostToHost:
		memcpy(p.h, p.d, bytes);
		break;
	case MemcpyKind::MemcpyDeviceToDevice:
		memcpy(p.d, p.h, bytes);
		break;
	}
}

_cuvahos void cuvaCopyToGPU(Linkp_void p, size_t bytes) {
	// no need to copy if linked pointer points to the same address
	// this is checked since on the cpu, linked pointers point to the 
	// same address by default.
	if (size_t(p.h) == size_t(p.d)) return;

	memcpy(p.d, p.h, bytes);
}
_cuvahos void cuvaCopyToCPU(Linkp_void p, size_t bytes) {
	// no need to copy if linked pointer points to the same address
	// this is checked since on the cpu, linked pointers point to the 
	// same address by default.
	if (size_t(p.h) == size_t(p.d)) return;

	memcpy(p.h, p.d, bytes);
}

_cuvahos void cuvaFree(void** p) {
	free(p);
}

_cuvahos void cuvaFree(Linkp_void p) {
	cuvaFree(p.h);
}


// threading has not been defined
#ifndef CUVAWORKS_THREAD

_cuvahos CuvaTarget getCuvaTarget() {
	return CuvaTarget::CuvaTarget_CPU;
}
_cuvahos const char* getCuvaTargetName() {
	return "CPU Device";
}

_cuvahos void cuvaSynchronize() {}

_cuvadev void getKernelID(KernelID* id) {}

_cuvahos void DispatchKernelAsync(Dim3 blocks, Dim3 threads, cuvakernel_t* kernel, void** args) {
	KernelID id;
	for (unsigned int bx = 0; bx < blocks.x; bx++) {
		for (unsigned int by = 0; by < blocks.y; by++) {
			for (unsigned int tx = 0; tx < threads.x; tx++) {
				for (unsigned int ty = 0; ty < threads.y; ty++) {
					id.i = tx + bx * threads.x;
					id.j = ty + by * threads.y;

					kernel(id, args);
				}
			}
		}
	}
}
_cuvahos void DispatchKernel(Dim3 blocks, Dim3 threads, cuvakernel_t* kernel, void** args) {
	DispatchKernelAsync(blocks, threads, kernel, args);
}

_cuvahos void cuvaClose() {}


// threading has been defined, use threadpool
#else

_cuvahos CuvaTarget getCuvaTarget() {
	return CuvaTarget::CuvaTarget_CPUThread;
}
_cuvahos const char* getCuvaTargetName() {
	return "CPU Threaded";
}

#include "cuvaworks_thread.h"

_cuvahos void cuvaSynchronize() {
	CuvaPool->Synchronize();
}

_cuvadev void getKernelID(KernelID* id) {}

_cuvahos void DispatchKernelAsync(Dim3 blocks, Dim3 threads, cuvakernel_t* kernel, void** args) {
	KernelID id;
	for (unsigned int bx = 0; bx < blocks.x; bx++) {
		for (unsigned int by = 0; by < blocks.y; by++) {
			for (unsigned int tx = 0; tx < threads.x; tx++) {
				for (unsigned int ty = 0; ty < threads.y; ty++) {
					id.i = tx + bx * threads.x;
					id.j = ty + by * threads.y;
					
					CuvaPool->AddKernel(kernel, id, args);
				}
			}
		}
	}
}

_cuvahos void DispatchKernel(Dim3 blocks, Dim3 threads, cuvakernel_t* kernel, void** args) {
	DispatchKernelAsync(blocks, threads, kernel, args);
	cuvaSynchronize();
}

_cuvahos void cuvaClose() {
	cuvaSynchronize();
}

#endif

#endif
