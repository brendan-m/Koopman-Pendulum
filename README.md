# Koopman Pendulum Tutorials: Learning non-linear dynamics with Koopman operator theory

# Table of contents
1. [Introduction](#introduction)
    1. [What is included in this repository?](#subparagraph11)
    2. [What is *not* included in this repository?](#subparagraph12)
       1. [The Deep Koopman framework](#subparagraph121)   
2. [Mathematics - What is Koopman operatory theory?](#paragraph1)
    1. [Sub paragraph](#subparagraph21)
3. [Applications](#paragraph2)
    1. [Pendulum (small angle approximation)](#subparagraph31)
    2. [Pendulum](#subparagraph32)
    3. [Double pendulum](#subparagraph33)
    4. [Soft pendulum](#subparagraph34)

## Introduction <a name="introduction"></a>
This repository is a collection of notebooks demonstrating analysis of non-linear pendulum dynamics with a Koopman operator theory based deep learning framework. This repository is **not** an extensive framework for learning (as such can likely **not** be run as is), but is rather a collection of analyses and evaluations to help support the understanding of Koopman operatory theory. 

This document was created to futher explore results published in [^fn0] (arxiv: [^fn00]) for some additional interesting use-cases. This document may be of interest to those studying non-linear dynamical systems, machine learning, or applications such as robotics.

### What is included in this repository?  <a name="subparagraph11"></a>

This repository contains a number of notebooks showing demonstrations of Koopman operatory theory in the context of learning pendulum dynamics. These notebooks demonstrate how learning dynamics in this manner can eludicate really interesting features of the underlying system. Specifically, in contrast to standard deep learning which often lack an intuitive internal physical interpretation, this approch characterises important physics-based structures as part of the learnt model, such as functional growths and oscillations. This is a very interesting way to provide physics-based explainations to prediction results, instead of relying purely on error metrics. 

### What is *not* included in this repository?  <a name="subparagraph12"></a>

#### The Deep Koopman framework itself<a name="subparagraph121"></a>

Learning the non-linear dynamics in a [Koopman operatory theory](#paragraph1) context involves identifying Koopman eigenfunctions using the DeepKoopman Network[^fn1]. The authors provide a deep learning framework (tensorflow) for this paper[^fn2], and a Keras version is avaliable by a third-party [^fn3]. These implementations will not be replicated here, and the interested reader is directory to those repositories for further details.


## Mathematics - What is Koopman operatory theory? <a name="paragraph1"></a>

A short summary of Koopman operator theory is presented here to help contextualise the notebooks. To avoid unnessecary replication, the reader is refered to the background section of [^fn00] for a more comprehensive overview and additional references. 


### Dynamical systems <a name="subparagraph21"></a>
In dynamical systems analysis, a discrete-time dynamical system on the state $\bf{x}$ is given as: $\bf{x}_{n+1} = \bf{F}(\bf{x}_x)$.
 
If $\bf{F}$ is linear, modelling these dynamics is relatively easy. However if $\bf{F}$ is non-linear, modelling becomes very difficult.

To address this, Koopman operator theory states we can always find coordinate transformations to map from the non-linear dynamics, to a latent linear dynamical system. Specifically, we describe the dynamics as the linear evolution of *measurement functions* of the non-linear state:

$\mathcal{K}g(\bf{x}_{n})$ = g(\bf{F}(\bf{x}_n))$ = g(\bf{x}_{n+1})$

where $\mathcal{K}$ is the Koopman operator, an infinite dimensional linear operator, acting on a measurement function $g$.

### Deep Koopman Network <a name="subparagraph22"></a>

An infinite dimensional linear operator is not particularly useful for a finite data system. However, the DeepKoopman Network[^fn1] addresses this by instead learning a set of spectral components of the Koopman operator, and using these as an intrinsic measurement coordinate system.

Specifically, $k$ different Koopman eigenfunctions $\bf{\phi}$ and eigenvalues $\lambda$ are learnt. These spectral components are a decomposition of the operator $\mathcal{K}$ satisfying $\mathcal{K}\phi_k = \phi_k \circ \bf{F} = \lambda_k \phi_k$. As such, we don't actually need to learn $\mathcal{K}$ just the eigenfunctions (and eigenvalues). 

These components are then used to create latent coordinates $\bf{y}=\phi(\bf{x})$ where the (linear) dynamics are learnt. Specifically, $\bf{y}$ is the transformation of the state $\bf{x}$ (which evolves non-linearly over time), into a latent state which evolves linearly over time.




## Applications <a name="paragraph2"></a>
Placeholder

### Pendulum (small angle approximation) <a name="subparagraph31"></a>
Placeholder

### Pendulum <a name="subparagraph32"></a>
Placeholder

### Double pendulum <a name="subparagraph33"></a>
Placeholder

### Soft pendulum <a name="subparagraph34"></a>
Placeholder

[^fn0]: Komeno, Naoto, Brendan Michael, Katharina Küchler, Edgar Anarossi, and Takamitsu Matsubara. "Deep Koopman with Control: Spectral Analysis of Soft Robot Dynamics." 2022 61st Annual Conference of the Society of Instrument and Control Engineers (SICE). IEEE, 2022. https://ieeexplore.ieee.org/abstract/document/9905758/
[^fn00]: https://arxiv.org/pdf/2210.07563.pdf
[^fn1]: Lusch, B., Kutz, J.N. & Brunton, S.L. Deep learning for universal linear embeddings of nonlinear dynamics. Nat Commun 9, 4950 (2018). https://doi.org/10.1038/s41467-018-07210-0
[^fn2]: https://github.com/BethanyL/DeepKoopman
[^fn3]: https://github.com/dykuang/Deep----Koopman
