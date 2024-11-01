# ISA-PINNs

\begin{abstract}

The improved self-adaptive PINNs (ISA-PINNs) are employed to numerically solve the high-order Sawada-Kotera (SK) and breaking soliton (BS) equations.
The method achieves high prediction accuracy, with relative $L_2$-norm errors in the range of $10^{-4}$ to $10^{-5}$ for both equations.
This represents a $1$-$2$ order of magnitude improvement in precision compared to previous results for the soliton molecule solutions of the SK equations.
Additionally, the simulation domain sizes are expanded by $2$-$3$ orders of magnitude while maintaining high precision for the two equations. 
Despite the larger domains, training time remains comparable due to the self-adaptive weight mechanism and automatic differentiation with the mesh-free method used in some simulations.
The ISA-PINNs model, developed with the open-source PyTorch framework, provides a significant speedup over conventional self-adaptive PINNs for solving the (2+1)-dimensional BS equation.
The robustness and generalization of ISA-PINNs are extensively validated across multiple solvers for both equations, with parameter scans covering domain sizes, initial and boundary conditions.
Boundary conditions can be eliminated from the loss function, and the model can still be trained to achieve similarly high prediction accuracies in both equation simulations.


This project builds on the research by Levi McClenny et al. [https://github.com/levimcclenny/SA-PINNs], enhancing training efficiency.

The current benchmark addresses the (2+1)-dimensional breaking soliton equation, with the full framework and additional examples planned for future updates.


