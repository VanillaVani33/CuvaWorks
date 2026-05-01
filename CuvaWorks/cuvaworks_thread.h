#pragma once

#include "cuvaworks_cuvakernel.h"

#ifdef CUVAWORKS_THREAD

#include <thread>
#include <stack>



class CuvaThreadPool {
private:

	std::stack<std::thread*> all_workers;


	CuvaThreadPool() {
		all_workers = std::stack<std::thread*>();
	}
public:


	static CuvaThreadPool* Start() {
		return new CuvaThreadPool();
	}

	void AddKernel(cuvakernel_t* kernel, KernelID id, void** args) {
		std::thread* th = new std::thread(kernel, id, args);
		all_workers.push(th);
	}

	void Synchronize() {
		std::thread* it;
		while (all_workers.size() > 0) {
			it = all_workers.top();
			all_workers.pop();
			if (it->joinable()) it->join();
		}
	}

};

CuvaThreadPool* CuvaPool = CuvaThreadPool::Start();


#endif