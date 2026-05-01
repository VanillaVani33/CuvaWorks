#pragma once
#include "cuvaworks_cuda_include.h"
#include "corecrt.h"
#include "corecrt_math.h"
#include <cstdio>


enum MemcpyKind {
	MemcpyHostToHost = 0,		// < Host->Host
	MemcpyHostToDevice = 1,     //< Host   -> Device 
	MemcpyDeviceToHost = 2,     //< Device -> Host 
	MemcpyDeviceToDevice = 3,   //< Device -> Device 
	MemcpyDefault = 4			//< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing 
};

enum CuvaTarget {
	CuvaTarget_CudaDevice = 0,
	CuvaTarget_CPU = 1,
	CuvaTarget_CPUThread = 2
};



#ifdef CUVAWORKS_CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define _cuvahos __host__
#define _cuvadev __device__
#define _hosdev _cuvahos _cuvadev
#define _devhos _cuvadev _cuvahos
#define _global __global__

typedef dim3 Dim3;

typedef uint1 UInt1;
typedef uint2 UInt2;
typedef uint3 UInt3;
typedef uint4 UInt4;

typedef int1 Int1;
typedef int2 Int2;
typedef int3 Int3;
typedef int4 Int4;

typedef float1 Float1;
typedef float2 Float2;
typedef float3 Float3;
typedef float4 Float4;




#else
// use for functions meant for the host device
#define _cuvahos
// use for functions meant for cuva devices
#define _cuvadev 
// use for functions meant for either host or cuva devices
#define _hosdev _cuvahos _cuvadev
#define _devhos _cuvadev _cuvahos
#define _global




struct UInt1 {
	unsigned int x;
};
struct UInt2 {
	unsigned int x, y;
};
struct UInt3 {
	unsigned int x, y, z;
};
struct UInt4 {
	unsigned int x, y, z, w;
};

struct Int1 {
	int x;
};
struct Int2 {
	int x, y;
};
struct Int3 {
	int x, y, z;
};
struct Int4 {
	int x, y, z, w;
};

struct Float1 {
	float x;
};
struct Float2 {
	float x, y;
};
struct Float3 {
	float x, y, z;
};
struct Float4 {
	float x, y, z, w;
};

struct Dim3 {
	unsigned int x, y, z;
	_hosdev constexpr Dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
	_hosdev constexpr Dim3(UInt3 v) : x(v.x), y(v.y), z(v.z) {}
	_hosdev constexpr operator UInt3(void) const { return UInt3{ x, y, z }; }
};


#endif





struct Linkp_void {
public:
	// host pointer
	void** h = 0;
	// device pointer
	void** d = 0;

	_hosdev Linkp_void() {}

	_hosdev Linkp_void(void** host_p, void** device_p) {
		h = host_p;
		d = device_p;
	}

	_hosdev Linkp_void(void** host_p) {
		h = host_p;
		d = host_p;
	}

	_hosdev Linkp_void operator+(size_t a) {
		return Linkp_void(h + a, d + a);
	}

	_hosdev Linkp_void operator-(size_t a) {
		return Linkp_void(h - a, d - a);
	}

	_hosdev size_t operator-(Linkp_void a) {
		return (h - a.h);
	}

};


template<typename T>
// establishes a link between cpu and gpu pointer
struct Linkp {
public:
	// host pointer
	T h;
	// device pointer
	T d;
	_hosdev Linkp() {}
	_hosdev Linkp(T host_pointer, T device_pointer) {
		h = host_pointer;
		d = device_pointer;
	}
	_hosdev Linkp(void** host_pointer, void** device_pointer) {
		h = (T)host_pointer;
		d = (T)device_pointer;
	}

	_hosdev Linkp(Linkp_void link) {
		h = (T)link.h;
		d = (T)link.d;
	}

	_hosdev Linkp(Linkp& linked_pointer) {
		h = linked_pointer.h;
		d = linked_pointer.d;
	}

	_hosdev Linkp(T p) {
		h = p;
		d = p;
	}

	_hosdev Linkp<T> operator+(size_t a) {
		return Linkp<T>(h + a, d + a);
	}

	_hosdev Linkp<T> operator-(size_t a) {
		return Linkp<T>(h - a, d - a);
	}

	_hosdev size_t operator-(Linkp<T> a) {
		return (h - a.h);
	}

	_hosdev operator Linkp_void() {
		return Linkp_void((void**)h, (void**)d);
	}
};


#define LinkedMemory_MemoryResolution 8ull


// Used to store mass amounts of data that can be transfered between cpu and gpu
// The idea is, all data will be stored side by side, so if i wanted to 
// transfer just one bit of data like variables data, i could, but if i wanted
// to transfer all data at once, i could as well. 
// its more efficient this way then to use cuda malloc for every array of data i need.
// This way, my malloc can allocate gpu memory by only speaking to the gpu once and using 
// some pointer arithmetic ! 
// instead of the hundreds of times to allocate memory for each array used
class MemoryBuffer {
private:
	// pointer to all the memory
	void** buffer = 0;
	// pointer to where the current memory is drawing from
	void** cursor = 0;


public:
	_hosdev MemoryBuffer() {
	}
	// block of memory
	_hosdev MemoryBuffer(void* PreAllocatedMemory) {
		buffer = (void**)PreAllocatedMemory;
		cursor = buffer;
	}

	// block of memory
	_hosdev MemoryBuffer(void** PreAllocatedMemory) {
		buffer = PreAllocatedMemory;
		cursor = buffer;
	}

	// allocates some number of bytes
	_hosdev void* malloc(size_t bytes) {
		// solves memory misallignment
		size_t rem = bytes % LinkedMemory_MemoryResolution;
		bytes += (LinkedMemory_MemoryResolution - rem) * (rem != 0);

		void** out = cursor;
		cursor += bytes;
		return out;
	}

	_hosdev void** getCursor() {
		return cursor;
	}

	_hosdev void** getBuffer() {
		return buffer;
	}

	_hosdev size_t MemoryUsed() {
		return cursor - buffer;
	}

	_hosdev size_t MemoryUsedKB() {
		return MemoryUsed() >> 10;
	}

	_hosdev size_t MemoryUsedMB() {
		return MemoryUsed() >> 20;
	}

	_hosdev size_t MemoryUsedGB() {
		return MemoryUsed() >> 30;
	}

	// returns where along the buffer the cursor lies, essentially 
	// giving the relative address to this memory
	_hosdev unsigned long int getCursorIndex() {
		return cursor - buffer;
	}

	// returns the index on the buffer that the given adress lies on
	_hosdev unsigned long int getIndexOf(void** address) {
		return address - buffer;
	}

	_hosdev void** getAddress(unsigned long int CursorIndex) {
		return buffer + CursorIndex;
	}
	_hosdev operator void* () { return buffer; }
};


class LinkedMemory {
private:
	Linkp_void buffer;
	size_t cursor;
public:
	size_t MemoryResolution;

	_hosdev LinkedMemory() {
		MemoryResolution = LinkedMemory_MemoryResolution;
	}

	_hosdev LinkedMemory(Linkp_void PreAllocatedMemory) {
		buffer = PreAllocatedMemory;
		cursor = 0;
		MemoryResolution = LinkedMemory_MemoryResolution;
	}


	_hosdev Linkp_void malloc(size_t bytes) {
		// solves memory misallignment
		size_t rem = bytes % MemoryResolution;
		bytes += (MemoryResolution - rem) * (rem != 0);

		Linkp_void out = buffer + cursor;
		cursor += bytes;
		return out;
	}

	_hosdev Linkp_void getCursor() {
		return buffer + cursor;
	}

	_hosdev Linkp_void getBuffer() {
		return buffer;
	}

	_hosdev size_t getCursorIndex() {
		return cursor;
	}

	_hosdev Linkp_void getAddress(size_t CursorIndex) {
		return buffer + CursorIndex;
	}

	operator Linkp_void() {
		return buffer;
	}

	_hosdev size_t MemoryUsed() {
		return cursor;
	}

	_hosdev size_t MemoryUsedKB() {
		return MemoryUsed() >> 10;
	}

	_hosdev size_t MemoryUsedMB() {
		return MemoryUsed() >> 20;
	}

	_hosdev size_t MemoryUsedGB() {
		return MemoryUsed() >> 30;
	}
};

