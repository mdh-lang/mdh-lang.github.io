---
layout: single
---

# Experimental Results

The experimental results presented in the following are described in detail [here](https://dl.acm.org/doi/10.1145/3665643).

## Performance Evaluation

Speedup of MDH (higher is better) over state-of-the-art approaches on GPUs and CPUs.

### Linear Algebra
![MDH Performance Experiments for Linear Algebra](/assets/images/exp_1.png)

### Stencils
![MDH Performance Experiments for Stencils](/assets/images/exp_2.png)

### Quantum Chemistry
![MDH Performance Experiments for Quantum Chemistry](/assets/images/exp_3.png)

### Data Mining
![MDH Performance Experiments for Data Mining](/assets/images/exp_4.png)

### Deep Learning
![MDH Performance Experiments for Deep Learning](/assets/images/exp_5.png)
![MDH Performance Experiments for Deep Learning](/assets/images/exp_6.png)


## Portability Evaluation

Our portability evaluation relies on the [Pennycook Metric](https://www.sciencedirect.com/science/article/pii/S0167739X17300559?casa_token=9ZtxQWqqghkAAAAA:XIXXlWXYjfwGE2KqY0gzuvmy8Mf_o4vtXEYAKY6dZECgDzWkrOwv_AMr2ObjkTn_jpty04kj_-0) (higher is better). A value close to `1` indicates high portability, and a value close to `0` indicates low portability.

### Portability Evaluation on GPUs+CPUs
![MDH Portability Experiments on GPUs and CPUs](/assets/images/eval_port_gpu_and_cpu.png)


### Portability Evaluation on either GPUs or CPUs

![MDH Portability Experiments on GPUs or CPUs for Linear Algebra](/assets/images/eval_port_gpu_or_cpu_la.png)

![MDH Portability Experiments on GPUs or CPUs for Stencils](/assets/images/eval_port_gpu_or_cpu_stencil.png)

![MDH Portability Experiments on GPUs or CPUs for Quantum Chemistry](/assets/images/eval_port_gpu_or_cpu_qc.png)

![MDH Portability Experiments on GPUs or CPUs for Probabilistic Record Linkage](/assets/images/eval_port_gpu_or_cpu_dl.png)