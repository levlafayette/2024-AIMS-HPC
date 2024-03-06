# General Structure of OpenACC

main()
{
 <CPU code>
 #pragma acc kernels
 //automatically runs on GPU
 {
  <GPU code>
 }
}

# Decomposition and Compilation

Examples exercises and solutions from Pawsey Supercomputing Centre.

1. Start an interactive job. Use a project ID that has gpgpu access.
$ sinteractive --partition=gpgpu -A hpcadmin --gres=gpu:p100:4
$ cd ~ ; cp -r /usr/local/common/OpenACC .
$ source /usr/local/module/spartan_old.sh
$ module load PGI/19.10-GCC-8.3.0-2.32

2. The Importance of Profiling

Excample here from ComputeCanada

$ cd ~/OpenACC/Profile
$ make

Check profile information.

3. Run serial code 

Examples here from Pawsey Supercomputing Centre.

$ cd ~/OpenACC/Exercise/exe1
$ make
$ time ./heat_eq_serial 

The output should be something like:

Stencil size: 2000 by 2000
Converged in 1000 iterations with an error of 0.0360

real	0m5.062s
user	0m5.051s
sys	0m0.008s

4. Identify parallel blocks.

PGI has inbuilt profiling tools. Nice!

time pgprof --cpu-profiling-scope instruction --cpu-profiling-mode top-down ./heat_eq_serial
Stencil size: 2000 by 2000
Converged in 1000 iterations with an error of 0.0360
======== CPU profiling result (top down):
Time(%)      Time  Name
 100.00%  5.15918s  ??? (0x400a49)
 17.42%  898.57ms    main (./heat_eq.c:94 0x493)
...

4. Introduce pragma statements

#pragma acc kernels directive [clause]
{
code region ..
}

cd ~/OpenACC/Solution/exe2

Note the compiler feedback! e.g.,

50, Loop not vectorized/parallelized: contains call
84, Loop is parallelizable
85, Loop is parallelizable
    Accelerator kernel generated
    Generating Tesla code
    84, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
    85, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */

etc

5. Run the code and profile again.

time pgprof --cpu-profiling-scope instruction --cpu-profiling-mode top-down ./heat_eq_acc_v2

Oh dear! Despite the fact regions have been parallised, the code is now *much* slower - due to to data movement. Most of the our time is spent in copying T_old and T_new from Host (CPU memory) to
Device (GPU memory).

6. Add Data construct pragmas, run the code again

cd ~/OpenACC/Solution/exe3

time pgprof --cpu-profiling-scope instruction --cpu-profiling-mode top-down ./heat_eq_acc_v2
