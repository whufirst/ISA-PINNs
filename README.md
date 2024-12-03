# ISA-PINNs

# Abstract

The improved self-adaptive PINNs (ISA-PINNs) are employed to numerically solve the high-order Sawada-Kotera (SK) and breaking soliton (BS) equations.
The method achieves high prediction accuracy, with relative $L_2$-norm errors in the range of $10^{-4}$ to $10^{-5}$ for both equations.
This represents a $1$-$2$ order of magnitude improvement in precision compared to previous results for the soliton molecule solutions of the SK equations.
Additionally, the simulation domain sizes are expanded by $2$-$3$ orders of magnitude while maintaining high precision for the two equations. 
Despite the larger domains, training time remains comparable due to the self-adaptive weight mechanism and automatic differentiation with the mesh-free method used in some simulations.
The ISA-PINNs model, developed with both TensorFlow and PyTorch frameworks, offers significantly improved precision compared to conventional self-adaptive PINNs for solving (2+1)-dimensional PDEs.
The robustness and generalization of ISA-PINNs are extensively validated across multiple solvers for both equations, with parameter scans covering domain sizes, initial and boundary conditions.
Boundary conditions can be eliminated from the loss function, and the model can still be trained to achieve similarly high prediction accuracies in both equation simulations.
The ISA-PINNs model is also applied to solve second- and third-order PDEs, demonstrating very high prediction accuracies as presented in the Appendix.


# Improvement

The ISA-PINNs model, implemented with both TensorFlow and PyTorch frameworks, achieves significantly enhanced precision compared to conventional SA-PINNs built on TensorFlow by McClenny et al. for solving ($2$+$1$)-dimensional PDEs.
The ISA-PINNs model implemented in PyTorch achieves significant speedup and considerably reduced memory consumption compared to its TensorFlow counterpart for solving the BS equation.

The current work addresses large domain simulations for solving the high-order (1+1)-dimensional SK and (2+1)-dimensional BS equations, where conventional PINNs models often fail.
The simulation domain sizes are expanded by 2-3 orders of magnitude, maintaining high precision for both the SK and BS equations, showcasing the robustness and effectiveness of the ISA-PINNs approach. 
For the soliton molecule solutions of the SK equations, the ISA-PINNs model achieves a 1-2 order of magnitude improvement in precision compared to previous results in small domain simulations.
By systematically scanning domain sizes, the training time and prediction precision remain comparable, attributed to the self-adaptive weight mechanism and the use of automatic differentiation with the mesh-free method in some simulations.

# Installation
ISA-PINNs implemented in PyTorch:

PyTorch version 2.0.0 or higher

CUDA versions 11 or 12

ISA-PINNs implemented in TensorFlow:

tensorflow version = 2.3.0

keras version = 2.4.3

The complete datasets are available at https://gitee.com/wilsonhu/isa-pinns, as GitHub imposes restrictions on data file sizes..
