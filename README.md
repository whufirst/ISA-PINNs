# ISA-PINNs

# Preprint

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5184258

# Citation
If you find this useful in your research, please consider citing:
    @article{hu5184258high,
      title={High-order partial differential equations solved by the improved self-adaptive PINNs},
      author={Hu, Wei and Dong, Chao},
      journal={Available at SSRN 5184258}
    }

# Improvement

1.  **Extension to 3D PDEs**: While the original SA-PINNs focused on 2D PDEs using TensorFlow, ISA-PINNs extend the self-adaptive algorithm to 3D PDEs, implemented in both TensorFlow and PyTorch.
This extension enables high-precision solutions for the BS and heat conduction (HC) equations using deep neural networks, overcoming training failures encountered with the original SA-PINNs.

2.  **Enhanced Accuracy and Larger Domains** : Prediction accuracies for the SK equation improve by $10$–$100$ times compared to prior studies using conventional PINNs.
Additionally, the computational domain is expanded by 2–3 orders of magnitude while maintaining high precision for the SK and BS equations, a capability not previously achieved with conventional PINNs.
Specifically for the BS equation, previous simulation domains ($x\times y\times t = [0,1]\times [0,1] \times [-1,1]$) were insufficient to capture the breaking soliton behavior, unlike our significantly larger domain of $x\times y\times t = [-40, 40]\times [-40, 40] \times [-4,4]$.

3.  **Efficiency and Scalability** : The ISA-PINNs PyTorch version offers significant speedup and reduced memory consumption compared to its TensorFlow counterpart for 3D PDEs.
The TensorFlow version, based on the original SA-PINNs, encounters out-of-memory issues when solving the 3D BS and HC equations with deep neural network models, a challenge efficiently addressed by the PyTorch implementation.

4.  **Improved Usability** : The PyTorch version integrates the L-BFGS algorithm as a direct function call within the framework, enhancing user-friendliness and compatibility across different software configurations.
In contrast, the original SA-PINNs are constrained to specific software setups, limiting adaptability to future updates.


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

[![YouTube Video](https://img.youtube.com/vi/qhd4ZoRVv5c/0.jpg)](https://www.youtube.com/watch?v=S9OBgS5tkms&ab_channel=WayHard)

Neural network architecture:

![Example Image](https://github.com/whufirst/ISA-PINNs/raw/main/isa-pinns.png)

<!-- https://github.com/whufirst/ISA-PINNs/raw/refs/heads/main/sa-sm1-animation.mp4 -->
<!-- https://gitee.com/wilsonhu/isa-pinns/raw/master/sa-sm1-animation.mp4 -->

<video width="640" height="360" controls>
  <source src="https://github.com/whufirst/ISA-PINNs/raw/refs/heads/main/sa-sm1-animation.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


