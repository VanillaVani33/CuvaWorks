READ ME ---------------------------------------------------------------------------- 
Overview: CuvaWorks is an interface designed to be implementable for both CUDA, 
and CPU programming simultaneously. It acts as an interface between the programmer 
and the machine so that a program meant to be run using an Nvidia Cuda Device 
could also run on the CPU, should the user decide to.

CHANGE FILE EXTENSION --------------------------------------------------------------

If you have Nvidia CUDA installed, copy all contents of your file containing "int main()" 
into a file with the ".cu" file extension.

If you dont have Nvidia CUDA installed, copy all contents of your file containing "int main()" 
into a file with the ".cpp" file extension. If it already has this extension, you are fine.

You HAVE to create a new file containing main with the new file extension, either ".cu" or ".cpp" 
You only need to make these changes once.

CHANGE CPU/GPU SETTINGS FOR CUVAWORKS ----------------------------------------------

After, follow the short instructions in the "\CuvaWorks\cuvaworks_cuda_include.h" to change to 
compile on your CUDA device, or on your CPU. You can change this as often as you'd like without 
changing the file extension.