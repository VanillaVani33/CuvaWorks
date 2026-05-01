#pragma once

// include cuvaworks
#include "CuvaWorks/cuvaworks.h"
// include cuvarand
#include "CuvaWorks/cuvarand.h"


// Structure holding kernel arguments, used as "void** arg"
struct kargs {
public:
	float* a;
	float* b;
	float* c;

	CuvaRand* cuva;
};



// Defining a Cuva Kernel that Initializes an instance of CuvaRand
_global void InitCuvaRand(KernelID id, void** arg) {
	// get indeces for kernel ID
	getKernelID(&id);

	// retrieve kernel arguments pointer
	kargs* args = (kargs*)arg;

	unsigned int i = id.i;
	unsigned int j = id.j;

	// Initialize an instance of CuvaRand
	initCuvarand(i, 0, 0, &args->cuva[i]);

}


// Defining a Cuva Kernel that adds two vectors, c = a + b
_global void AddAB(KernelID id, void** arg) {
	// get indeces for kernel ID
	getKernelID(&id);

	// retrieve kernel arguments pointer
	kargs* args = (kargs*)arg;

	unsigned int i = id.i;
	unsigned int j = id.j;

	// Add a and b and offset a random ammount
	args->c[i] = args->a[i] + args->b[i] + uniCuvarand(&args->cuva[i], -1, -1);
}

int main() {

	// memory to preallocate (10 kb)
	size_t prealloc = 1024ul * 10ull;

	// size of vectors
	unsigned int sample = 50;

	// linked memory buffer, for host and device
	LinkedMemory mem = cuvaMallocLink(prealloc);

	// a vector
	Linkp<float*> a = mem.malloc(sample * sizeof(float));
	// b vector
	Linkp<float*> b = mem.malloc(sample * sizeof(float));
	// c vector
	Linkp<float*> c = mem.malloc(sample * sizeof(float));

	// kernel arguments
	Linkp<kargs*> args = mem.malloc(sizeof(kargs));

	// Cuvarand Instances
	Linkp<CuvaRand*> cuva = mem.malloc(sample * sizeof(CuvaRand));

	// initialize vector values
	for (unsigned int i = 0; i < sample; i++) {
		a.h[i] = i;
		b.h[i] = 2 * i + i;
	}

	// set the kernel arguments
	args.h->a = a.d;
	args.h->b = b.d;
	args.h->c = c.d;
	args.h->cuva = cuva.d;

	// copy all data to gpu
	cuvaCopyToGPU(mem, prealloc);

	// initialize Cuvarand instances
	DispatchKernel(1, sample, InitCuvaRand, (void**)args.d);
	// add the vectors 
	DispatchKernel(1, sample, AddAB, (void**)args.d);

	// copy all data too cpu
	cuvaCopyToCPU(mem, prealloc);


	// print results
	for (unsigned int i = 0; i < sample; i++) {
		printf(" a[%i] + b[%i] = %f + %f = %f\n", i, i, a.h[i], b.h[i], c.h[i]);
	}

	// close cuva session
	cuvaFree(mem);
	cuvaClose();

	return 67;
}