# Green HPC

[Repository view](https://github.com/EPCCed/archer2-gpu-hipc/)

[Pages view](https://epcced.github.io/archer2-gpu-hipc/)

This short course will provide an introduction to GPU computing with HIP aimed at scientific application programmers wishing to develop their own software.
The course will give a background on the difference between CPU and GPU architectures as a prelude to introductory exercises in HIP programming. The course
will discuss the execution of kernels, memory management, and shared memory operations. Common performance issues are discussed and their solution addressed.

The course will go on to consider execution of independent streams, and the execution of work composed as a collection of dependent tasks expressed as a graph.
Device management and details of device to device data transfer will be covered for situations where more than one GPU device is available.

The course will not discuss programming with compiler directives, but does provide a concrete basis for an understanding of the underlying principles of the HIP model,
which is useful for programmers ultimately wishing to make use of OpenMP or OpenACC (or indeed other models). The course will not consider graphics programming,
nor will it consider machine learning packages.

Note, this course is also appropriate for those wishing to use NVIDIA GPUs via the CUDA API, although we will not specifically use CUDA.

Attendees must be able to program in C or C++ (course examples and exercises will limit themselves to C++). A familiarity with threaded programming models would be useful,
but no previous knowledge of GPU programming is required.

See below for links to lectures and exercises.

* [Lectures](lectures/)
* [Practical exercises](exercises/)
