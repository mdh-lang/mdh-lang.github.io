---
layout: splash
---

![](/assets/images/mdh_banner.png)

# Overview

The approach of [Multi-Dimensional Homomorphisms (MDH)](https://dl.acm.org/doi/10.1145/3665643) is an algebraic formalism for systematically reasoning about *de-composition* and *re-composition* strategies of data-parallel computations (such as linear algebra routines and stencil computations) for the memory and core hierarchies of state-of-the-art parallel architectures (GPUs, multi-core CPU, multi-device and multi-node systems, etc).

The MDH approach (formally) introduces:

1. [*High-Level Program Representation*](/mdh_hl) *(Contribution 1)* that enables the user conveniently implementing data-parallel computations, agnostic from hardware and optimization details;
2. [*Low-Level Program Representation*](/mdh_ll) *(Contribution 2)* that expresses device- and data-optimized de- and re-composition strategies of computations;
3. [*Lowering Process*](/mdh_hl_to_ll) *(Contribution 3)* that fully automatically lowers a data-parallel computation expressed in its high-level program representation to an optimized instance in its low-level representation, based on concepts from automatic performance optimization (a.k.a. *auto-tuning*), using the [Auto-Tuning Framework (ATF)](https://www.atf-tuner.org).

The MDH's low-level representation is designed such that [Code Generation](/mdh_code_gen) from it (e.g., in [OpenMP](https://www.openmp.org) for CPUs, [CUDA](https://developer.nvidia.com/cuda-toolkit) for NVIDIA GPUs, or [OpenCL](https://www.khronos.org/opencl/) for multiple kinds of architectures) becomes straightforward.

![Overview of MDH Approach](/assets/images/overview.png)

Our [Experiments](/experiments) report encouraging results on GPUs and CPUs for MDH as compared to state-of-practice approaches, including NVIDIA [cuBLAS](https://developer.nvidia.com/cublas)/[cuDNN](https://developer.nvidia.com/cudnn) and Intel [oneMKL](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-0/intel-oneapi-math-kernel-library-onemkl.html)/[oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html) which are hand-optimized libraries provided by vendors.

### Ultimate MDH Goals

- **Performance**: achieve performance competitive to hand-optimized solutions
- **Portability**: target any kind of parallel architecture
- **Productivity**: free user from hardware and optimization details

<br>
# Getting Started
*(Our implementation of MDH will be open sourced soon on GitHub)*


<br>
# Code Examples

From the following code examples, our MDH compiler generates fully automatically device- and data-optimized, executable program code, e.g., in OpenMP for CPUs, CUDA for NVIDIA GPUs, or OpenCL for multiple kinds of architectures.

## MDH's [Python](https://www.python.org)-Based User Interface

<div class="tab-container">

{% tabs code_examples_python %}

{% tab code_examples_python MatVec %}

<hr>

*Matrix-Vector Multiplication (MatVec)* expressed in MDH's high-level program representation:

~~~ python
def matvec(T: ScalarType, I: int, K: int):
    @mdh( out( w = Buffer[T, [I]]                        ) ,
          inp( M = Buffer[T, [I, K]], v = Buffer[T, [K]] ) )
    def mdh_matvec():
        def mul(out, inp):
            out['w'] = inp['M'] * inp['v']

        def scalar_plus(res, lhs, rhs):
            res['w'] = lhs['w'] + rhs['w']

        return (
            out_view[T]( w = [lambda i, k: (i)] ),
              md_hom[I, K]( mul, ( CC, PW(scalar_plus) ) ),
                inp_view[T, T]( M = [lambda i, k: (i, k)] ,
                                v = [lambda i, k: (k)   ] )
        )
~~~

The above defined `matvec` function is used as follows:

~~~ python
# MatVec on 1024x1024-sized input matrix and 1024-sized vector (both containing fp32 values)
matvec__fp32__1024_1024 = matvec( fp32, 1024,1024 )

# ... (CUDA host code: create CUDA context, CUDA buffers for "M","v", "w", etc.)

# Get MDH "CUDA Module" for MatVec (using ATF-tuned optimizations)
cuda__matvec__fp32__1024_1024 = matvec__fp32__1024_1024.get_module( CUDA(), pyATF( CUDARuntimeProfiler(), evaluations(1000) ) )

# MDH CUDA Module: compile & load CUDA code
a100_cuda__matvec__fp32__1024_1024 = cuda__matvec__fp32__1024_1024.compile( arch='compute_80' )

# MDH CUDA Module: run MatVec on M,v to obtain w
a100_cuda__matvec__fp32__1024_1024.run( w,M,v )

# MDH CUDA Module: destroy module
a100_cuda__matvec__fp32__1024_1024.destroy()

# ... (CUDA host code: destroying CUDA context, freeing CUDA buffers, etc.)
~~~

{% endtab %}

{% tab code_examples_python Jacobi1D %}

<hr>

*Jacobi-1D (Jacobi1D)* expressed in MDH's high-level program representation:

~~~ python
def jacobi1d(T: ScalarType, I: int):
    @mdh( out( y = Buffer[T, [I]]     ) ,
          inp( x = Buffer[T, [I + 2]] ) )
    def mdh_jacobi1d():
        def f_j1d(out, inp):
            out['y'] = (inp['x', 1] + inp['x', 2] + inp['x', 3]) / 3.0

        return (
            out_view[T]( y = [lambda i: (i)] ),
              md_hom[I]( f_j1d, ( CC ) ),
                inp_view[T]( x = [lambda i: (i + 0),
                                  lambda i: (i + 1),
                                  lambda i: (i + 2)] )
        )
~~~

The above defined `jacobi1d` function is used as follows:

~~~ python
# Jacobi1D on a 1024-sized input vector (containing fp32 values)
jacobi1d__fp32_1024 = jacobi1d( fp32, 1024 )

# ... (CUDA host code: create CUDA context, CUDA buffers for "x", "y", etc.)

# Get MDH "CUDA Module" for Jacobi1D (using ATF-tuned optimizations)
cuda__jacobi1d__fp32_1024 = jacobi1d__fp32_1024.get_module( CUDA(), pyATF( CUDARuntimeProfiler(), evaluations(1000) ) )

# MDH CUDA Module: compile & load CUDA code
a100_cuda__jacobi1d__fp32_1024 = cuda__jacobi1d__fp32_1024.compile( arch='compute_80' )

# MDH CUDA Module: run Jacobi1D on x to obtain y
a100_cuda__jacobi1d__fp32_1024.run( y,x )

# MDH CUDA Module: destroy module
a100_cuda__jacobi1d__fp32_1024.destroy()

# ... (CUDA host code: destroying CUDA context, freeing CUDA buffers, etc.)
~~~

{% endtab %}

{% endtabs %}

</div>

## MDH's [MLIR](http://mlir.llvm.org)-Based User Interface

<div class="tab-container">

{% tabs code_examples_mlir %}

{% tab code_examples_mlir MatVec %}

<hr>

~~~ llvm
func.func @main()
{
  %M = memref.alloc() : memref<128x64xf32>
  %v = memref.alloc() : memref<64xf32>

  %w = mdh.compute "mdh_matvec"
  {
    inp_view =
    [
      [ affine_map<( i,k ) -> ( i,k )> ],
      [ affine_map<( i,k ) -> ( k )  > ]
    ],

    md_hom =
    {
      scalar_func = @mul,
      combine_ops = [ "cc", ["pw",@add] ]
    },

    out_view =
    [
      [ affine_map<( i,k ) -> ( i )> ]
    ]
  }
  {
    inp_types = [ f32, f32 ],
    mda_size  = [ 128,64 ],
    out_types = [ f32 ]
  }( %M,%v ) :
   ( memref<128x64xf32> ,
     memref<64xf32>     ) -> memref<128xf32>

  return
}
~~~

Here, functions `@mul` and `@add` are straightforward, user-defined functions for computing *scalar multiplication* or *scalar addition*, respectively (both not shown for brevity).
Functions `cc` and `pw` are pre-implemented *combine operators* for computing *concatenation* (`cc`) or *point-wise operations* (`pw`), respectively.

{% endtab %}

{% tab code_examples_mlir Jacobi1D %}

<hr>

~~~ llvm
func.func @main()
{
  %x = memref.alloc() : memref<64xf32>

  %y = mdh.compute "mdh_jacobi1d"
  {
    inp_view =
    [
      [ affine_map<( i ) -> ( i+0 )> ,
        affine_map<( i ) -> ( i+1 )> ,
        affine_map<( i ) -> ( i+2 )> ]
    ],

    md_hom =
    {
      scalar_func = @jacobi,
      combine_ops = [ "cc" ]
    },

    out_view =
    [
      [ affine_map<( i ) -> ( i )> ]
    ]
  }
  {
    inp_types = [ f32 ],
    mda_size  = [ 62 ],
    out_types = [ f32 ]
  }( %x ) : (memref<64xf32>) -> (memref<62xf32>)

  return
}
~~~

Here, function `@jacobi` is the Jacobi-specific computation (not shown for brevity), and `cc` is a pre-implemented *combine operator* for computing *concatenation*.

{% endtab %}

{% endtabs %}

</div>


## Automatic Parallelization & Optimization

Additionally, MDH supports as inputs -- as an alternative to DSL programs in MDH's high-level programming interface (shown above) -- also straightforward (annotated) sequential program code.
For our *MatVec* example, our Python-based input code is of the following form:

~~~ python
def matvec(T: ScalarType, I: int, K: int):
    @mdh( out( w = Buffer[T, [I]]                        ) ,
          inp( M = Buffer[T, [I, K]], v = Buffer[T, [K]] ) ,
          combine_ops = ( CC, PW(scalar_plus) )            )
    def mdh_matvec(w, M, v):
        for i in range(I):
            for k in range(K):
                w[i] = M[i, k] * v[k]
~~~

This program is completely equivalent to the DSL-based MDH program for MatVec shown above and used exactly the same:

~~~ python
# MatVec on 1024x1024-sized input matrix and 1024-sized vector (both containing fp32 values)
matvec__fp32__1024_1024 = matvec( fp32, 1024,1024 )

# ... (CUDA host code: create CUDA context, CUDA buffers for "M","v", "w", etc.)

# Get MDH "CUDA Module" for MatVec (using ATF-tuned optimizations)
cuda__matvec__fp32__1024_1024 = matvec__fp32__1024_1024.get_module( CUDA(), pyATF( CUDARuntimeProfiler(), evaluations(1000) ) )

# MDH CUDA Module: compile & load CUDA code
a100_cuda__matvec__fp32__1024_1024 = cuda__matvec__fp32__1024_1024.compile( arch='compute_80' )

# MDH CUDA Module: run MatVec on M,v to obtain w
a100_cuda__matvec__fp32__1024_1024.run( w,M,v )

# MDH CUDA Module: destroy module
a100_cuda__matvec__fp32__1024_1024.destroy()

# ... (CUDA host code: destroying CUDA context, freeing CUDA buffers, etc.)
~~~

## Schedule-Based Optimization Process

MDH optionally allows incorporating expert knowledge into the optimization process, using its [scheduling language](https://dl.acm.org/doi/abs/10.1145/3578360.3580269).
By incorporating the user into the optimization process, we enable two major advantages over the standard MDH workflow:
  1. better optimization, as an auto-tuning system might not always make the same high-quality optimization decisions as a human expert
  2. faster auto-tuning, as some (or even all) optimization decisions might be made by the expert user and thus are not left to the costly auto-tuner


~~~ python
(An example scheduling program follows soon)
~~~

<br>
# Publications

1.  A. Rasch \\
    [(De/Re)-Composition of Data-Parallel Computations via Multi-Dimensional Homomorphisms](https://dl.acm.org/doi/10.1145/3665643) \\
    *ACM Transactions on Programming Languages and Systems (TOPLAS 2024)*\\
    <a href="../assets/files/publications/toplas24/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](../assets/files/publications/toplas24/paper.pdf)
    <a href="assets/files/publications/pldi24/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/pldi24/slides.pdf)

3.  A. Rasch \\
    [Full Version: (De/Re)-Composition of Data-Parallel Computations via Multi-Dimensional Homomorphisms](https://arxiv.org/abs/2405.05118) \\
    *arXiv 2024*\\
    <a href="../assets/files/publications/arxiv24/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](../assets/files/publications/arxiv24/paper.pdf)
    <a href="assets/files/publications/pldi24/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/pldi24/slides.pdf)

5.  A. Rasch, R. Schulze, D. Shabalin, A. Elster, S. Gorlatch, M. Hall \\
    [(De/Re)-Compositions Expressed Systematically via MDH-Based Schedules](https://dl.acm.org/doi/abs/10.1145/3578360.3580269) \\
    *ACM SIGPLAN International Conference on Compiler Construction (CC 2023)* \\
    <a href="assets/files/publications/cc23/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/cc23/paper.pdf)
    <a href="assets/files/publications/cc23/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/cc23/slides.pdf)

6.  A. Rasch, R. Schulze, and S. Gorlatch \\
    [Generating Portable High-Performance Code via Multi-Dimensional Homomorphisms](https://ieeexplore.ieee.org/abstract/document/8891668) \\
    *International Conference on Parallel Architectures and Compilation Techniques (PACT 2019)*\\
    <a href="assets/files/publications/pact19/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/pact19/paper.pdf)
    <a href="assets/files/publications/pact19/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/pact19/slides.pdf)

7.  A. Rasch, R. Schulze, M. Gorus, J. Hiller, S. Bartholomäus, S. Gorlatch \\
    [High-Performance Probabilistic Record Linkage via Multi-Dimensional Homomorphisms](https://dl.acm.org/doi/abs/10.1145/3297280.3297330) \\
    *ACM/SIGAPP Symposium On Applied Computing (SAC 2018)*\\
    <a href="assets/files/publications/sac18/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/sac18/paper.pdf)
    <a href="assets/files/publications/sac18/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/sac18/slides.pdf)

8.  A. Rasch, S. Gorlatch \\
    [Multi-Dimensional Homomorphisms and Their Implementation in OpenCL](https://link.springer.com/article/10.1007/s10766-017-0508-z) \\
    *International Journal of Parallel Programming (IJPP 2018)*\\
    <a href="assets/files/publications/ijpp18/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/ijpp18/paper.pdf)


### WIP/Short Papers & Talks

1.  A. Rasch, R. Schulze, Jens Hunloh, Lars Hunloh \\
    [Code Generation & Optimization for Deep-Learning Computations via Multi-Dimensional Homomorphisms](https://www.c4ml.org) \\
    *Compilers for Machine Learning (C4ML 2024), (lightning talk)* \\
    <a href="assets/files/publications/c4ml24/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Slides](assets/files/publications/c4ml24/slides.pdf)
    <a href="assets/files/publications/c4ml24/poster_1.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [1. Poster](assets/files/publications/c4ml24/poster_1.pdf)
    <a href="assets/files/publications/c4ml24/poster_2.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [2. Poster](assets/files/publications/c4ml24/poster_2.pdf)

2.  A. Rasch, R. Schulze, S. Gorlatch \\
    [Array Programming via Multi-Dimensional Homomorphisms](assets/files/publications/pldi23/paper.pdf) \\
    *ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI 2023), (WIP paper)* \\
    <a href="assets/files/publications/pldi23/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/pldi23/paper.pdf)
    <a href="assets/files/publications/pldi23/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/pldi23/slides.pdf)
    <a href="https://www.youtube.com/watch?v=FAwgO86b6oo"><i class="fas fa-video" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Talk](https://www.youtube.com/watch?v=FAwgO86b6oo)

3.  R. Schulze, A. Rasch, S. Gorlatch \\
    [Code Generation & Optimization for Deep-Learning Computations on GPUs via Multi-Dimensional Homomorphisms](https://sc21.supercomputing.org/proceedings/tech_poster/) \\
    *International Conference for High Performance Computing, Networking, Storage and Analysis (SC 2021), **Best Poster Finalist**, (short paper)*\\
    <a href="assets/files/publications/sc21/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/sc21/paper.pdf)
    <a href="assets/files/publications/sc21/poster.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Poster](assets/files/publications/sc21/poster.pdf)
    <a href="assets/files/publications/sc21/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/sc21/slides.pdf)
    <a href="https://www.youtube.com/watch?v=nBJCc6TGUrg"><i class="fas fa-video" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Talk](https://www.youtube.com/watch?v=nBJCc6TGUrg)

4.  A. Rasch, R. Schulze, S. Gorlatch \\
    [Using MLIR for Multi-Dimensional Homomorphisms](https://groups.google.com/a/tensorflow.org/g/mlir/c/CxFj0UKBBRw) \\
    *Google SIG MLIR Open Design Meeting 2020, (invited talk)*\\
    <a href="assets/files/publications/google20/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Slides](assets/files/publications/google20/slides.pdf)
    <a href="https://www.youtube.com/watch?v=RQR_9tHscMI"><i class="fas fa-video" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Talk](https://www.youtube.com/watch?v=RQR_9tHscMI)

5.  A. Rasch, S. Gorlatch \\
    [md_stencil: High-Performance Stencil Computations on CPU and GPU via Multi-Dimensional Homomorphisms](https://pact20.cc.gatech.edu/acm-src/) \\
    *International Conference on Parallel Architectures and Compilation Techniques (PACT 2020), (SRC -- **Gold Winner**)*\\
    <a href="assets/files/publications/pact20/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/pact20/paper.pdf)
    <a href="assets/files/publications/pact20/poster.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Poster](assets/files/publications/pact20/poster.pdf)
    <a href="assets/files/publications/pact20/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/pact20/slides.pdf)
    <a href="https://www.youtube.com/watch?v=DGWjHudmkUc&t=4s"><i class="fas fa-video" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Talk](https://www.youtube.com/watch?v=DGWjHudmkUc&t=4s)

6.  A. Rasch, S. Gorlatch \\
    [md_poly: A Performance-Portable Polyhedral Compiler Based on Multi-Dimensional Homomorphisms](https://impact-workshop.org/impact2020/) \\
    *IEEE/ACM International Symposium on Code Generation and Optimization (CGO 2020), (SRC -- **Gold Winner**)*\\
    <a href="assets/files/publications/cgo20/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/cgo20/paper.pdf)
    <a href="assets/files/publications/cgo20/poster.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Poster](assets/files/publications/cgo20/poster.pdf)
    <a href="assets/files/publications/cgo20/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/cgo20/slides.pdf)

7.  A. Rasch, R. Schulze, S. Gorlatch \\
    [Performance, Portability, and Productivity for Data-Parallel Applications on Multi- and Many-Core Architectures](https://sc19.supercomputing.org/presentation/index-id=drs106&sess=sess264.html) \\
    *International Conference for High Performance Computing, Networking, Storage and Analysis (SC 2019), (doctoral showcase)*\\
    <a href="assets/files/publications/sc19/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/publications/sc19/paper.pdf)
    <a href="assets/files/publications/sc19/poster.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Poster](assets/files/publications/sc19/poster.pdf)
    <a href="assets/files/publications/sc19/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/publications/sc19/slides.pdf)


<!--
1.  A. Rasch, R. Schulze, S. Gorlatch \\
    [md_poly: A Performance-Portable Polyhedral Compiler Based on Multi-Dimensional Homomorphisms](https://impact-workshop.org/impact2020/) \\
    *International Workshop on Polyhedral Compilation Techniques (IMPACT 2020), (WIP paper)*\\
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](TODO)
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Poster](TODO)
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](TODO)
-->

<!--
1.  A. Rasch \\
    [Performance, Portability, and Productivity for Data-parallel Applications on Multi- and Many-core Architectures](https://dl.acm.org/doi/abs/10.1145/3359061.3361072) \\
    *ACM SIGPLAN International Conference on Systems, Programming, Languages, and Applications: Software for Humanity (SPLASH 2019), (short paper)*\\
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](TODO)
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](TODO)
-->

<!--
1.  A. Rasch, R. Schulze, S. Gorlatch. \\
    [Expressing Hierarchical Code Optimizations via MDH-Based Schedules](https://hipar.ng-analytics.com/wp-content/uploads/2022/11/hipar22_paper4.png) \\
    *Hierarchical Parallelism for Exascale Computing (HiPar@SC 2022), (WIP paper)*\\
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](TODO)
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](TODO)
-->

<!--
1.  A. Rasch, R. Schulze, S. Gorlatch \\
    [Developing High-Performance, Portable OpenCL Code via Multi-Dimensional Homomorphisms](https://dl.acm.org/doi/abs/10.1145/3318170.3318171) \\
    *International Workshop on OpenCL (IWOCL 2019), (extended abstract)*\\
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](TODO)
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](TODO)
-->

<!--
1.  A. Rasch, R. Schulze, S. Gorlatch
    [Portable Parallel Performance via Multi-Dimensional Homomorphisms](https://sc18.supercomputing.org/proceedings/tech_poster/poster_files/post118s2-file3.png) \\
    *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC) (2018), (short paper)*\\
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](TODO)
    <a href="TODO"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](TODO)
-->

<br>
# Citations

Please use the following citations, when referring to MDH's:

1. *Formalism & Design*
~~~
@article{10.1145/3665643,
  author = {Rasch, Ari},
  title = {(De/Re)-Composition of Data-Parallel Computations via Multi-Dimensional Homomorphisms},
  year = {2024},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  issn = {0164-0925},
  url = {https://doi.org/10.1145/3665643},
  doi = {10.1145/3665643},
  journal = {ACM Trans. Program. Lang. Syst.},
  month = {may}}
~~~

2. *Scheduling Language*
~~~
@inproceedings{10.1145/3578360.3580269,
author = {Rasch, Ari and Schulze, Richard and Shabalin, Denys and Elster, Anne and Gorlatch, Sergei and Hall, Mary},
title = {(De/Re)-Compositions Expressed Systematically via MDH-Based Schedules},
year = {2023},
isbn = {9798400700880},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3578360.3580269},
doi = {10.1145/3578360.3580269},
booktitle = {Proceedings of the 32nd ACM SIGPLAN International Conference on Compiler Construction},
pages = {61-72},
numpages = {12},
keywords = {GPU, scheduling languages, deep learning, CPU},
location = {Montr\'{e}al, QC, Canada},
series = {CC 2023}
}
~~~

3. *Automatic Parallelization and Optimization*
~~~
@INPROCEEDINGS{mdpoly,
  author={Rasch, Ari and Schulze, Richard and Gorlatch, Sergei},
  booktitle={Proceedings of the International Workshop on Polyhedral Compilation Techniques (IMPACT 2020)},
  title={\texttt{md\_poly}: A Performance-Portable Polyhedral Compiler based on Multi-Dimensional Homomorphisms},
  year={2020},
  volume={},
  number={},
  pages={1-4}}
~~~

<br>
# Contact

<div class="card_container">
  <div class="card">
    <div class="card_content">
      <a href="https://www.arirasch.net" style="color: black"><img src="assets/images/ari.JPG" alt="Avatar"></a>
      <a href="https://www.arirasch.net" style="color: black"><h4><b>Ari Rasch</b></h4></a>
      <table>
        <tr><td>Focus:</td><td>Formalism</td></tr>
        <tr><td>Affiliation:</td><td><a href="https://www.uni-muenster.de/en/">University of Münster</a></td></tr>
        <tr><td>Email:</td><td><a href="mailto:a.rasch@uni-muenster.de?cc=r.schulze@uni-muenster.de">a.rasch@uni-muenster.de</a></td></tr>
        <tr><td>Website:</td><td><a href="https://www.arirasch.net">arirasch.net</a></td></tr>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card_content">
      <a href="https://www.richardschulze.net" style="color: black"><img src="assets/images/richard.PNG" alt="Avatar"></a>
      <a href="https://www.richardschulze.net" style="color: black"><h4><b>Richard Schulze</b></h4></a>
      <table>
        <tr><td>Focus:</td><td>Implementation</td></tr>
        <tr><td>Affiliation:</td><td><a href="https://www.uni-muenster.de/en/">University of Münster</a></td></tr>
        <tr><td>Email:</td><td><a href="mailto:r.schulze@uni-muenster.de?cc=a.rasch@uni-muenster.de">r.schulze@uni-muenster.de</a></td></tr>
        <tr><td>Website:</td><td><a href="https://www.richardschulze.net">richardschulze.net</a></td></tr>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card_content">
      <a href="TODO" style="color: black"><img src="assets/images/lars.jpg" alt="Avatar"></a>
      <a href="TODO" style="color: black"><h4><b>Lars Hunloh</b></h4></a>
      <table>
        <tr><td>Focus:</td><td>MDH in <a href="https://mlir.llvm.org">MLIR</a></td></tr>
        <tr><td>Affiliation:</td><td><a href="https://www.uni-muenster.de/en/">University of Münster</a></td></tr>
        <tr><td>Email:</td><td><a href="mailto:l.hunloh@uni-muenster.de?cc=a.rasch@uni-muenster.de,r.schulze@uni-muenster.de">l.hunloh@uni-muenster.de</a></td></tr>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card_content">
      <a href="TODO" style="color: black"><img src="assets/images/jens.jpg" alt="Avatar"></a>
      <a href="TODO" style="color: black"><h4><b>Jens Hunloh</b></h4></a>
      <table>
        <tr><td>Focus:</td><td>MDH in <a href="https://mlir.llvm.org">MLIR</a></td></tr>
        <tr><td>Affiliation:</td><td><a href="https://www.uni-muenster.de/en/">University of Münster</a></td></tr>
        <tr><td>Email:</td><td><a href="mailto:j.hunloh@uni-muenster.de?cc=a.rasch@uni-muenster.de,r.schulze@uni-muenster.de">j.hunloh@uni-muenster.de</a></td></tr>
      </table>
    </div>
  </div>
</div>

<br>

You can also find us on <a href="https://discord.gg/9x5HHWmFXV"><img src="assets/images/discord-logo.svg" alt="Discord" style="height: 20px"> Discord</a>.
