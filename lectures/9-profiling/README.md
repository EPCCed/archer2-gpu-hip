template: titleslide
# Profiling



---
# Profiling

So far we have used the simple text-based reporting tool for profiling, `nvprof`.

NVIDIA also provide more sophisticated tools, Nsight Systems and
NSight Compute for when you need to profile specific kernels.

- https://developer.nvidia.com/nsight-systems

- https://developer.nvidia.com/nsight-compute

The first port-of-call should always be Nsight Systems as the measurements
from that tool will tell you which kernels are worth profiling.



---
# Nsight Systems

Compile and run as usual but prefix the executable with `nsys`.

```bash
nsys profile -o systems ./a.out
```

Running your code in this way should produce a file called `systems.nsys-rep`,
which can be read by the Nsight Systems UI.

The usual mode of operation is to copy the report file back to your local machine.

See the link below for all `nsys` command line options.<br>
[https://docs.nvidia.com/nsight-systems/UserGuide/index.html#command-line-options](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#command-line-options)



---
# Adding NVTX markers

A bare profile shows CUDA API (host) activity and device activity (memory copies and kernels).
It can be useful however to identify particular sections of code.

NVTX (NVIDIA Toolkit extension) markers can be added to host code as follows.

```c
#include "nvToolsExt.h"

...

nvtxRangeId_t id = nvtxRangeStartA("MY ASCII LABEL");

/* ... code of interest ... */

nvtxRangeEnd(id);
```

Simply add a range start and end either side of the code your wish to profile and then
recompile (you may need to add `-lnvToolsExt`).

A coloured bar representing the code of interest should appear in the profile visualisation.



---
# Nsight Compute

Use `ncu` to profile particular kernels.

```bash
ncu -o default ./a.out
```

The above run should produce a `default.ncu-rep` file which you can load within the Nsight Compute UI.
There are alternative ways to run `ncu` that collect further levels of information via additional passes of the kernel(s).

```bash
ncu --set detailed -o detailed ./a.out
ncu --set full -o full ./a.out
```



---
# Nsight Compute

Use `ncu` to profile particular kernels.

```bash
ncu -o default ./a.out
```

The above run should produce a `default.ncu-rep` file which you can load within the Nsight Compute UI.
There are alternative ways to run `ncu` that collect further levels of information via additional passes of the kernel(s).

```bash
ncu --set detailed -o detailed ./a.out
ncu --set full -o full ./a.out
```

![:thumb](Running `ncu` can be time-consuming, and so, care should be taken to limit the data collected,
e.g. by restricting the profiling to some subset of the calls of a single kernel.<br><br>
See the link below for all `ncu` command line options.<br>
[https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options))



---
# <span style="color:red">Exercise: Profiling</span>

For the simple matrix operation ([exercise_dger.hip.cpp](../../exercises/9-profiling/1-profiling)) we developed earlier, try to run first Nsight Systems
and then Nsight Compute with the various options.

Have a go at adding some NVTX markers to highlight a region of host code.



---
# <span style="color:red">Next Lecture</span>

<br>
## [Streams](../10-streams)