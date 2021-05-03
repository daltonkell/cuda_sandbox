===============
Vector Addition
===============

This short CUDA program is a simple intro to allocating CUDA device memory, copying data to and
from the host machine, and calling device kernels. Specifically, I was interested in checking out
how fast the intrinsic addition functions were compared to "vanilla C++ addition" (which is probably
optimized in the compiler anyway, but I have yet to know for sure).

NVCC two-stage compilation is done through a quick BASH script, build.bash.

Program output:

```
dkell@dkell-desktop:~/cuda_sandbox/vecAdd$ ./vecAdd 
Just sanity checking host arrays...
hostA first element: 0.840188
hostA last element: 0.126484
hostB first element: 0.394383
hostB last element: 0.649627
CUDA kernel launch with 98 blocks of 256 threads
Kernel launched with vanilla addition
Finished in ... 63856 nanoseconds
CUDA kernel launch with 98 blocks of 256 threads
Kernel launched with __fadd_rn instrinsic addition
Finished in ... 57292 nanoseconds
```

I haven't done any rigorous testing yet, but it seems that the instrinsic `__fadd_rn` which rounds
to the nearest is always faster (though imperceptibly with such a simple calculation). It's also accurate
enough to pass the very rudimentary absolute value check:

```
    // Verify that the result vector is correct
    for (int i = 0; i < nElements; ++i)
    {
        if (fabs(hostA[i] + hostB[i] - hostC[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
```

It would be interesting to change this error threshold to see where it fails. According to the
CUDA documentation, the function is IEEE Compliant, which I don't knopw exactly what that means.
