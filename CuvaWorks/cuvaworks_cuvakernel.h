#pragma once
#include "cuvaworks_types.h"


// cuda kernel prelimiters
struct KernelID {

public:
	// unique x-index of the current thread being executed
	unsigned int i;
	// unique y-index of the current thread being executed
	unsigned int j;

	_hosdev KernelID() {}	

	_hosdev KernelID(KernelID& id) {
		i = id.i;
		j = id.j;
	}
	_hosdev KernelID(unsigned int p) {
		i = p;
		j = p;
	}
};

// function pointer for a cuva kernel
// KernelID: The ID of a kernel 
// void**:   Arguments you may pass onto the kernel
typedef void cuvakernel_t(KernelID, void**);



