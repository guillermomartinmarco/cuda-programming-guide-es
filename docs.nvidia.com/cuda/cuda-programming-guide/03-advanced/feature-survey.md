# 3.5. A Tour of CUDA Features

Sections 1-3 of this programming guide have introduced CUDA and GPU programming, covering foundational topics both conceptually and in simple code examples. The sections describing specific CUDA features in part 4 of this guide assume knowledge of the concepts covered in sections 1-3 of this guide.

CUDA has many features which apply to different problems. Not all of them will be applicable to every use case. This chapter serves to introduce each of these features and describe its intended use and the problems it may help solve. Features are coarsely sorted into categories by the type of problem they are intended to solve. Some features, such as CUDA graphs, could fit into more than one category.

[Section 4](https://docs.nvidia.com/cuda/cuda-programming-guide/part4.html#cuda-features) covers these CUDA features in more complete detail.

## 3.5.1. Improving Kernel Performance [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#improving-kernel-performance "Link to this heading")

The features outlined in this section are all intended to aid kernel developers to maximize the performance of their kernels.

### 3.5.1.1. Asynchronous Barriers [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#asynchronous-barriers "Link to this heading")

[Asynchronous barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers) were introduced in [Section 3.2.4.2](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers) and allow for more nuanced control over synchronization between threads. Asynchronous barriers separate the arrival and the wait of a barrier. This allows applications to perform work that does not depend on the barrier while waiting for other threads to arrive. Asynchronous barriers can be specified for different [thread scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-thread-scopes). Full details of asynchronous barriers are found in [Section 4.10](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers).

### 3.5.1.2. Asynchronous Data Copies and the Tensor Memory Accelerator (TMA) [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#asynchronous-data-copies-and-the-tensor-memory-accelerator-tma "Link to this heading")

[Asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies) in the context of CUDA kernel code refers to the ability to move data between shared memory and GPU DRAM while still carrying out computations. This should not be confused with asynchronous memory copies between the CPU and GPU. This feature makes use of asynchronous barriers. [Section 4.12](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies) covers the use of asynchronous copies in detail.

### 3.5.1.3. Pipelines [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#pipelines "Link to this heading")

[Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines) are a mechanism for staging work and coordinating multi-buffer producer–consumer patterns, commonly used to overlap compute with [asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies). [Section 4.11](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines) has details and examples of using pipelines in CUDA.

### 3.5.1.4. Work Stealing with Cluster Launch Control [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#work-stealing-with-cluster-launch-control "Link to this heading")

Work stealing is a technique for maintaining utilization in uneven workloads where workers that have completed their work can ‘steal’ tasks from other workers. Cluster launch control, a feature introduced in compute capability 10.0 (Blackwell), gives kernels direct control over in-flight block scheduling so they can adapt to uneven workloads in real time. A thread block can cancel the launch of another thread block or cluster that has not yet started, claim its index, and immediately begin executing the stolen work. This work-stealing flow keeps SMs busy and cuts idle time under irregular data or runtime variation—delivering finer-grained load balancing without relying on the hardware scheduler alone.

[Section 4.13](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#cluster-launch-control) provides details on how to use this feature.

## 3.5.2. Improving Latencies [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#improving-latencies "Link to this heading")

The features outlined in this section share a common theme of aiming to reduce some type of latency, though the type of latency being addressed differs between the different features. By and large they are focused on latencies at the kernel launch level or higher. GPU memory access latency within a kernel is not one of the latencies considered here.

### 3.5.2.1. Green Contexts [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#green-contexts "Link to this heading")

[Green contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-contexts), also called _execution contexts_, is the name given to a CUDA feature which enables a program to create [CUDA contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api-context) which will execute work only on a subset of the SMs of a GPU. By default, the thread blocks of a kernel launch are dispatched to any SM within the GPU which can fulfill the resource requirements of the kernel. There are a large number of factors which can affect which SMs can execute a thread block, including but not necessarily limited to: shared memory use, register use, use of clusters, and total number of threads in the thread block.

Execution contexts allow a kernel to be launched into a specially created context which further limits the number of SMs available to execute the kernel. Importantly, when a program creates a green context which uses some set of SMs, other contexts on the GPU will not schedule thread blocks onto the SMs allocated to the green context. This includes the primary context, which is the default context used by the CUDA runtime. This allows these SMs to be reserved for workloads which are high priority or latency-sensitive.

[Section 4.6](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-contexts) gives full details on the use of green contexts.

### 3.5.2.2. Locality Domains [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#locality-domains "Link to this heading")

A [locality domains](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/locality-domains.html#locality-domains) is a subset of GPU memory and SMs. On GPUs with more than one locality domain, it can be beneficial to allocate memory within a specific locality domain and then execute kernels using that memory in a green context which has only SMs in the same locality domain.

Locality domains are a performance feature: CUDA code which does not localize memory or compute will still execute correctly.

[Section 1.2.3.2](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-locality-domains) introduces the concept behind locality domains and [Section 4.7](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/locality-domains.html#locality-domains) gives more information about how to determine how many locality domains a GPU has, how to allocate memory into a specific locality domain, and how to create a green context with SMs only from that same locality domain.

### 3.5.2.3. Stream-Ordered Memory Allocation [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#stream-ordered-memory-allocation "Link to this heading")

The [stream-ordered memory allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator) allows programs to sequence allocation and freeing of GPU memory into a [CUDA stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams). Unlike `cudaMalloc` and `cudaFree` which execute immediately, `cudaMallocAsync` and `cudaFreeAsync` inserts a memory allocation or free operation into a CUDA stream. [Section 4.3](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator) covers all the details of these APIs.

### 3.5.2.4. CUDA Graphs [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#cuda-graphs "Link to this heading")

[CUDA graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs) enable an application to specify a sequence of CUDA operations such as kernel launches or memory copies and the dependencies between these operations so that they can be executed efficiently on the GPU. Similar behavior can be attained by using [CUDA streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams), and indeed one of the mechanisms for creating a graph is called [stream capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture), which enables the operations on a stream to be recorded into a CUDA graph. Graphs can also be created using the [CUDA graphs API](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-graph-apis).

Once a graph has been created, it can be instantiated and executed many times. This is useful for specifying workloads that will be repeated. Graphs offer some performance benefits in reducing CPU launch costs associated with invoking CUDA operations as well as enabling optimizations only available when the whole workload is specified in advance.

[Section 4.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs) describes and demonstrates how to use CUDA graphs.

### 3.5.2.5. Programmatic Dependent Launch [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#programmatic-dependent-launch "Link to this heading")

[Programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization) is a CUDA feature which allows a dependent kernel, i.e. a kernel which depends on the output of a prior kernel, to begin execution before the primary kernel on which it depends has completed. The dependent kernel can execute setup code and unrelated work up until it requires data from the primary kernel and block there. The primary kernel can signal when the data required by the dependent kernel is ready, which will release the dependent kernel to continue executing. This enables some overlap between the kernels which can help keep GPU utilization high while minimizing the latency of the critical data path. [Section 4.5](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization) covers programmatic dependent launch.

### 3.5.2.6. Lazy Loading [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#lazy-loading "Link to this heading")

[Lazy loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#lazy-loading) is a feature which allows control over how the JIT compiler operates at application startup. Applications which have many kernels which need to be JIT compiled from PTX to cubin may experience long startup times if all kernels are JIT compiled as part of application startup. The default behavior is that modules are not compiled until they are needed. This can be changed by the use of [environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-environment-variables), as detailed in [Section 4.8](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#lazy-loading).

## 3.5.3. Functionality Features [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#functionality-features "Link to this heading")

The features described here share a common trait that they are meant to enable additional capabilities or functionality.

### 3.5.3.1. Extended GPU Memory [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#extended-gpu-memory "Link to this heading")

[Extended GPU memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#extended-gpu-memory) is a feature available in NVLink-C2C connected systems that enables efficient access to all memory within the system from within a GPU. EGM is covered in detail in [Section 4.19](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#extended-gpu-memory).

### 3.5.3.2. Dynamic Parallelism [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#dynamic-parallelism "Link to this heading")

CUDA applications most commonly launch kernels from code running on the CPU. It is also possible to create new kernel invocations from a kernel running on the GPU. This feature is referred to as [CUDA dynamic parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism). [Section 4.20](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism) covers the details of creating new GPU kernel launches from code running on the GPU.

## 3.5.4. CUDA Interoperability [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#cuda-interoperability "Link to this heading")

### 3.5.4.1. CUDA Interoperability with other APIs [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#cuda-interoperability-with-other-apis "Link to this heading")

There are other mechanisms than CUDA for running code on GPUs. The application GPUs were originally built to accelerate, computer graphics, uses its own set of APIs such as Direct3D and Vulkan. Applications may wish to use one of the graphics APIs for 3D rendering while performing computations in CUDA. CUDA provides mechanisms for exchanging data stored on the GPU between the CUDA contexts and the GPU contexts used by the 3D APIs. For example, an application may perform a simulation using CUDA, and then use a 3D API to create visualizations of the results. This is achieved by making some buffers readable and/or writeable from both CUDA and the graphics API.

The same mechanisms which allow sharing of buffers with graphics APIs are also used to share buffers with communications mechanisms which can enable rapid, direct GPU-to-GPU communication within multi-node environments.

[Section 4.21](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#cuda-interoperability) describes how CUDA interoperates with other GPU APIs and how to share data between CUDA and other APIs, providing specific examples for a number of different APIs.

### 3.5.4.2. Interprocess Communication [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#interprocess-communication "Link to this heading")

For very large computations, it is common to use multiple GPUs together to make use of more memory and more compute resources working together on a problem. Within a single system, or node in cluster computing terminology, multiple GPUs can be used in a single host process. This is described in [Section 3.4](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-gpu-introduction).

It is also common to use separate host processes spanning either a single computer or multiple computers. When multiple processes are working together, communication between them is known as interprocess communication. CUDA interprocess communication (CUDA IPC) provides mechanisms to share GPU buffers between different processes. [Section 4.16](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication) explains and demonstrates how CUDA IPC can be used to coordinate and communicate between different host processes.

## 3.5.5. Fine-Grained Control [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#fine-grained-control "Link to this heading")

### 3.5.5.1. Virtual Memory Management [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#virtual-memory-management "Link to this heading")

As mentioned in [Section 2.6.1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-virtual-address-space), all GPUs in a system, along with the CPU memory, share a single unified virtual address space. Most applications can use the default memory management provided by CUDA without the need to change its behavior. However, [the CUDA driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api) provides advanced and detailed controls over the layout of this virtual memory space for those that need it. This is mostly applicable for controlling the behavior of buffers when sharing between GPUs both within and across multiple systems.

[Section 4.17](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management) covers the controls offered by the CUDA driver API, how they work and when a developer may find them advantageous.

### 3.5.5.2. Compute Fabric Transport [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#compute-fabric-transport "Link to this heading")

[Section 4.18](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/compute-fabric-transport.html#compute-fabric-transport) describes a resource-centric model for GPU-initiated data movement over an NVLink fabric, in which a peer is named by a logical endpoint id and offset rather than by a mapped virtual address. The feature targets developers writing communication primitives, applications should prefer the [NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) or [NVSHMEM](https://docs.nvidia.com/nvshmem/api/index.html) when either matches their communication pattern.

### 3.5.5.3. Driver Entry Point Access [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#driver-entry-point-access "Link to this heading")

[Driver entry point access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#driver-entry-point-access) refers to the ability, starting in CUDA 11.3, to retrieve function pointers to the CUDA Driver and CUDA Runtime APIs. It also allows developers to retrieve function pointers for specific variants of driver functions, and to access driver functions from drivers newer than those available in the CUDA toolkit. [Section 4.22](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#driver-entry-point-access) covers driver entry point access.

### 3.5.5.4. Error Log Management [\#](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html\#error-log-management "Link to this heading")

[Error log management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management) provides utilities for handling and logging errors from CUDA APIs. Setting a single environment variable `CUDA_LOG_FILE` enables capturing CUDA errors directly to stderr, stdout, or a file. Error log management also enables applications to register a callback which is triggered when CUDA encounters an error. [Section 4.9](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management) provides more details on error log management.

On this page