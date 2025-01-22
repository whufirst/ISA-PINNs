# ISA-PINNs

The improved SA-PINNs (ISA-PINNs) are developed to numerically solve the high-order Sawada-Kotera (SK) and breaking soliton (BS) equations, achieving high prediction accuracy with relative $L_2$-norm errors in the range of $10^{-4}$ to $10^{-5}$.

# Improvement

1.  **Extension to 3D PDEs**: While the original SA-PINNs focused on 2D PDEs using TensorFlow, the ISA-PINNs extend the self-adaptive algorithm to 3D PDEs, implemented in both TensorFlow and PyTorch.
This extension achieves high precision for the BS and heat conduction (HC) equations.

2.  **Efficiency and Scalability** : The ISA-PINNs PyTorch version offers significant speedup and reduced memory consumption compared to its TensorFlow counterpart for 3D PDEs.
The TensorFlow version, based on the original SA-PINNs, encounters out-of-memory issues when solving the 3D BS and HC equations with deep neural network models, a challenge efficiently addressed by the PyTorch implementation.

3.  **Improved Usability** : The PyTorch version integrates the L-BFGS algorithm as a direct function call within the framework, enhancing user-friendliness and compatibility across different software configurations.
In contrast, the original SA-PINNs are constrained to specific software setups, limiting adaptability to future updates.

4.  **Enhanced Accuracy and Larger Domains** : For the SK equation, ISA-PINNs employ the FP64 data type to avoid NaN issues encountered in the original SA-PINNs.
This approach improves prediction accuracy by $10$–$100$ times compared to prior studies.
Moreover, the domain sizes for the SK and BS equations in this study are approximately $2$–$3$ orders of magnitude larger than those in earlier works.

The ISA-PINNs PyTorch codes include implementations for the BS and Burgers' equations.

The ISA-PINNs TensorFlow template is provided with the 3D HC equation and can also be utilized to compare training speed and memory consumption with the PyTorch version for solving BS and HC equations.

For the 3D HC equation featuring both Gaussian and periodic source terms, ISA-PINNs PyTorch version employs a deep neural network (7 hidden layers with 64 neurons each) to achieve high prediction precision.
In contrast, the ISA-PINNs TensorFlow version, based on the original SA-PINNs, faces memory inefficiency issues with deep neural networks and is unable to train effectively.
Moreover, the TensorFlow version fails to achieve high prediction accuracy when using a shallow neural network.

# Installation
ISA-PINNs implemented in PyTorch:

PyTorch version 2.0.0 or higher

CUDA versions 11 or 12

ISA-PINNs implemented in TensorFlow:

tensorflow version = 2.3.0

keras version = 2.4.3

The complete datasets are available at https://gitee.com/wilsonhu/isa-pinns, as GitHub imposes restrictions on data file sizes.

This video demonstrates the evolution of self-adaptive weights for the SM1 case.

[![YouTube Video](https://img.youtube.com/vi/qhd4ZoRVv5c/0.jpg)](https://www.youtube.com/watch?v=qhd4ZoRVv5c)

<!-- https://github.com/whufirst/ISA-PINNs/raw/refs/heads/main/sa-sm1-animation.mp4 -->
<!-- https://gitee.com/wilsonhu/isa-pinns/raw/master/sa-sm1-animation.mp4 -->

<video width="640" height="360" controls>
  <source src="https://github.com/whufirst/ISA-PINNs/raw/refs/heads/main/sa-sm1-animation.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If the improvements to SA-PINNs are helpful to your work, please consider citing one of the preprint papers listed below.

```
@article{hu5056812high,
  title={High-Order Partial Differential Equations Solved by the Improved Self-Adaptive Pinns},
  author={Hu, Wei and Zheng, Shaolong and Dong, Chao and Chen, Miao and Fei, Jin-Xi and Gao, Ruozhou},
  journal={Available at SSRN 5056812}
}
```
