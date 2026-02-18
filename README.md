## Project ideas to use the library for
1. Train a neural net on the solution to the equation to simulate molecular dynamics faster ex.PhysicsX

Base the work partialy on these 3 papers: idea train on existing equations so the net predicts the solution faster than the numerical method within a suitable error

1. Fourier Neural Operator — Li et al. (2020), "Fourier Neural Operator for Parametric Partial Differential Equations" (arXiv:2010.08895). This is the foundation for operator learning. The math is clean and the architecture maps directly to CUDA kernels (FFT + pointwise transforms). Start here.
2. NequIP — Batzner et al. (2022), "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials" (Nature Communications 13, 2453). This is the key paper for molecular systems specifically — it shows how to build neural nets that respect rotational/translational symmetry (E(3) equivariance) so you don't waste capacity learning physics your architecture should enforce for free. Directly relevant to your stat mech setting.
3. DeePMD-kit — Wang et al. (2018), "DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics" (Computer Physics Communications 228, 178-184). This is the most practical reference — it's an actual CUDA-optimized implementation of ML potentials for molecular dynamics. You can study their kernel design decisions as a blueprint for your own custom forward/backward passes.
