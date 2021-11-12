<!-- .slide: data-state="no-toc-progress" -->

Institute of Neural Information Processing | Ulm University

<h1 class="no-toc-progress">Instance-based Learning</h1>

Dr. Sebastian Gottwald 

---

## 1. Motivation

--

### Basic Idea


* __Given:__ Data $\mathcal D = \\{(x_i,y_i)\\}_{i=1}^N$ of patterns $x_i\in \mathcal X$ and targets $y_i\in \mathcal Y$.
* __Goal:__ Predict the target $y$ of a new pattern $x\in \mathcal X$.

* Need __additional structure__ to compare $x$ to the known information:
  * Parametrized models use a __loss function__ defined on $\mathcal Y$ to __compare outputs__ during training (and use the error to adapt the model parameters, e.g. gradient-descent).

  * (most) instance-based models use a __kernel function__ defined on $\mathcal X$ to __compare inputs__


<div style="margin-top: 50px;"></div>


<div class="comment">
<b>Note:</b> Here, a kernel is a non-negative function $k:\mathcal X\times\mathcal X \to\mathbb R^+, (x,x')\mapsto k(x,x')$ that is used to measure similarity of $x$ and $x'$, but often kernels are required to satisfy additional constraints such as positive definiteness or symmetry (we will see this later).
</div>

###

The term _kernel_ is overused:
* __Linear Algebra:__ A subset of the domain of a linear map that is mapped to $0$.
* __Statistics:__ A non-negative function that appears as part of a probability distribution or density (e.g. $e^{-(x-\mu)^2/(2\sigma^2)}$)
* __Analysis:__ An integrable function that appears in integral operators.


--



### Example 1: Kernel regression (Nadaraya-Watson model)

See [notes](pdfs/LS2_InstanceBasedLearning_Nadaraya-Watson.pdf) for details. 
* Consider dataset $\mathcal D = \\{(x_n, y_n)\\}_{n=1}^N$ as a sample from a joint distribution $\mathbb P(X,Y)$.
* Approximate the joint density by $\ p(x,y) := \frac{1}{N} \sum_{n=1}^N f(x-x_n) g(y-y_n)$.  
* Define the regression function to be $y(x) := \mathbb E_{p(Y|X=x)}[Y]$.
* A short calculation shows that
$$y(x) = \frac{\int y \ p(x,y) dy}{\int p(x,y) dy} = \cdots = \sum_n y_n \underbrace{\frac{f(x-x_n)}{\sum_m f(x-x_m)}}_{=:\ k_D(x,x_n)} =\sum_n y_n k_D(x,x_n) 
$$ 


<div class="comment">
<b>Note:</b> $k_D(x,x_n)\in [0,1]$ and $\sum_n k_D(x,x_n) = 1$, i.e. $q(n):=k_D(x,x_n)$ can be viewed as a probability distribution over the patterns $x_n$, favoring those $x_n$ that are considered "similar" to $x$, and thus $y(x)$ is the average of $y_n$ wrt. $q$.
</div>



--


### Inner product kernels

If a kernel $k$ can be written as an __inner product__ on some space $\mathcal H$, a so-called _feature space_, in the sense that 
$$
k(x,x') = \langle \phi(x), \phi(x')\rangle_\mathcal H
$$
for some mapping $\phi:\mathcal X\to\mathcal H$, a so-called __feature map__, then $k$ is called an __inner product kernel__ (often the prefix _inner product_ is dropped!). 

The image $\phi(x)\in\mathcal H$ of a pattern $x$ under $\phi$ is then called a _feature vector_, and its components are called _features_.

--

### Why use inner products?

Inner products can serve as __similarity measures__ in vector spaces: For $x,x' \in \mathbb R^d$, we have

$$
\langle x, x'\rangle  = \\|x\\| \ \\|x'\\| \ \cos \phi(x,x')
$$ 
where $\phi(x,x')$ is the angle between $x$ and $x'$. 

* If $\\|x\\|=\\|x'\\| = 1$, then $\langle x,x'\rangle \in [-1,1]$ is maximal if $\cos \phi(x,x') = 1$, i.e. if $x$ and $x'$ point in the same direction.

* Hence, $\langle x,x'\rangle$ is a good similarity measure, if the length (or _magnitude_) of the vectors are not informative. 

* E.g. when the vectors are normalized in some way, so that only the proportions between the features are relevent, not their total value.

* Or when having decision hyperplanes through the origin, so that only the angle is used as a criterion.

<!-- Thus, if we keep the length (or _magnitude_) of the two vectors fixed, then $\langle x, x'\rangle$ becomes maximal if $x$ and $x'$ point into the same direction.

For feature vectors, **direction is usually more informative than magnitude**, especially since one often normalizes the vectors in some way.
 -->
<div style="margin-bottom: 20px;"></div>


<!-- <div class="comment">
<b>Note:</b> In fact, $\frac{1}{\|x\|}\langle x,x'\rangle$ is the length of the projection of $x'$ along the direction of $x$.
</div>
 -->
<!-- <div style="margin-bottom: 10px;"></div> -->

<div class="comment">
<b>Note:</b> The angle between two vectors can also be defined in arbitrary/infinite dimensional inner product spaces by the above equality (due to the <i>Cauchy-Schwarz inequality</i>, c.f. next section).
</div>



--


### Why use feature maps?

* If $\mathcal X$ is a set without vector space structure (e.g. words), then a feature map $\phi$ __embeds__ $\mathcal X$ into an inner product space, where the inner product allows to measure similarity.

* Even if $\mathcal X$ is already a vector space with an inner product, it might not measure the right notion of similarity for a given problem.


<div style="margin-bottom: 30px; margin-top: 30px;" class="fragment">

### Why use inner product kernels?
The features might live in a very high (maybe even infinite) dimensional space, but the kernel could have a closed form that does not require the explicit calculation of the features.
</div>

<div class="fragment">

### rule of thumb: feature maps vs kernels 

* __Kernels__ have an advantage when the feature space is high dimensional 

* __Feature maps__ are better if the number of samples is very large

</div>

--

### Some feature maps and their kernels

<br>

|Feature map  | $\Rightarrow$ | Kernel |
| --- | --- | --- | 
|$\phi:\mathbb R^d \to \mathbb R^d, x\mapsto x$ | | $k(x,x') = \langle x,x'\rangle_{\mathbb R^d}$ $= \mathbf{x}^T \mathbf{x'}$ |
|$\phi:\mathbb R^d \to \mathbb R^{d^2}, x\mapsto (x_ix_j)_{i,j=1}^d$ | | $k(x,x') = \big(\langle x, x'\rangle_{\mathbb R^d}\big)^2$ |
|$\phi: \\{0,1,2\\} \to [0,1], x\mapsto p(x)$ | | $k(x,x') = p(x) p(x')$ |
|$\phi:2^\Omega \to L^\infty(\Omega), A \mapsto \mathbb{1}_A {-} P(A)$ | | $k(A,B) = P(A\cap B) - P(A)P(B)$  |


<div style="margin-top: 50px;"></div>

__The converse:__ How do we know that a kernel, e.g. $\ f(x,x') = e^{-||x-x'||^2}$, is an inner product kernel (i.e. can be written as an inner product of $\phi(x)$ and $\phi(x')$ for some $\phi$)? 

__Answer:__ Hilbert space theory (next section).



---

## 2. Reproducing Kernel Hilbert Spaces

--

### Vector spaces

A _vector space_ (or _linear space_) $V$ consists of elements $v$ (called _vectors_) that can be added ($v+w\in V$, if $v,w\in V$) and  multiplied by scalars ($\alpha v \in V$ if $\alpha\in \mathbb R$, $v\in V$). Examples include 

* __Euclidean spaces__ $\mathbb R^d$: $\alpha x + y \in \mathbb R^d$, where $(\alpha x + y)_i := \alpha x_i + y_i$ (elementwise)

* __Sequence spaces:__ $\alpha (x_n)_n + (y_n)_n := (\alpha x_n + y_n)_n$ (elementwise), <br>e.g. bounded sequences $\ell^\infty$, summable sequences $\ell^1$, square-summable sequences $\ell^2,\dots$.

* __Function spaces:__ $(\alpha f+g)(x) := \alpha f(x) + g(x)$ (pointwise), <br>e.g. continuous functions $C([a,b])$ on an interval $[a,b]$, continuously differentiable functions $C^1((a,b))$, square integrable functions $L^2(\mathbb R)$ on $\mathbb R$, $\dots$



<div style="margin-top: 20px;"></div>


<div class="comment">
<b>Note:</b> For the purpose of this lecture, we assume that $L^2(\mathbb R)$ consists of functions. Rigorously, one has to consider equivalence classes of functions that are equal <i>almost everywhere</i>, which means that $f,g\in L^2(\mathbb R)$ are considered the same even if $f(x) \not= g(x)$ on a set of measure $0$ ($A\subset \mathbb R$ has measure $0$ if $\int_A dx = 0$, e.g. $A=\{x\}$ $\forall x\in\mathbb R$). 
</div>

--

### Inner product spaces

An _inner product space_ is a vector space $V$ together with an _inner product_ $\langle \cdot,\cdot\rangle$, which (in the real case) is a function $\langle \cdot,\cdot\rangle: V\times V\to\mathbb R$ that is __symmetric__, __linear__ in both entries, and __positive definite__ ($\langle x,x\rangle >0$ if $x\not = 0$). 

Examples:

* Euclidean spaces $(\mathbb R^d,\langle \cdot,\cdot\rangle_{\mathbb R^d})$, where $\langle x,y\rangle_{\mathbb R^d} = \sum_{i=1}^d x_i y_i$ for $x,y\in\mathbb R^d$.

* Sequence spaces, e.g. $(\ell^2, \langle\cdot,\cdot\rangle_{\ell^2})$, where $\langle x, y\rangle_{\ell^2} = \sum_{i=1}^\infty x_i y_i$ for $x,y\in\ell^2$.

* Function spaces, e.g. $(L^2(\mathbb R), \langle\cdot,\cdot\rangle_{L^2})$, where $\langle f, g\rangle_{L^2} = \int_{\mathbb R} f(x) g(x) \ dx$.


###

<div style="margin-top: 50px;"></div>


<div class="comment">
<b>Note:</b> 
<ul>
    <li>An inner product space $V$ is an example of a <i>normed space</i> with norm $\|v\| := \sqrt{\langle v, v\rangle}$ for all $v\in V$. A norm $\|v\|$ measures the <i>length</i> of a vector $v$, and therefore introduces a notion of <i>distance</i> by $\|v-w\|$. </li>
    <li>An important result is the <i>Cauchy-Schwarz inequality</i>: $|\langle v,w\rangle| \leq \|v\| \|w\|$.</li>
</ul>
</div>

--

### Induced Norm 

An inner product space $V$ is an example of a __normed space__ with norm $\\|v\\| = \sqrt{\langle v, v\rangle}$ for all  $v\in V$. A norm measures the _length_ of a vector $v$, and therefore introduces a notion of _distance_ into $V$ by $d(v,w) = \\|v-w\\|$ (a so-called _metric_), which, in turn, implies a notion of convergence (a _topology_). 

Examples of __norms__ that are __induced by inner products__:

* Euclidean norm: $\ \\|x\\| = \sqrt{\sum_{i=1}^d x_i^2} = \sqrt{\langle x, x\rangle_{\mathbb R^d}}$ for $x\in \mathbb R^d$ (analogous for $\ell^2$)

* Function norm in $L^2$: $\ \\|f\\| = \sqrt{\int |f(x)|^2 dx} = \langle f,f\rangle_{L^2}$ for $f\in L^2(\mathbb R)$.

Examples of __norms__ that do __not come from inner products__:

* $\ell^p$ and $L^p$ norms for $p\not=2$: $\\|x\\|_p  = \left( \sum_i |x_i|^p \right)^{1/p}$, $\\|f\\|_p  = \left( \int |f(x)|^p dx \right)^{1/p}$

* the supremum norms $\\|x\\|_\infty = \sup_i |x_i|$ and $\\|f\\|_\infty = \sup_x |f(x)|$.  


--


### Cauchy-Schwarz inequality

__Theorem:__ _For all elements $v,w\in V$ of an inner product space $V$, we have_
$$|\langle v,w\rangle| \leq \\|v\\| \\|w\\|.$$


* In $\mathbb R^d$, this can be seen as a consequence of $\langle x,y\rangle = \\|x\\| \\|y\\| \cos \theta$. In fact, it justifies the __definition of an angle__ between elements of arbitrary inner product spaces.

* It implies the __triangle inequality__, $\\|v+w\\| \leq \\|v\\|+ \\|w\\|$, in any inner product space (see Exercises). 

* It is very useful to show implications like $x,y\in \ell^2 \Rightarrow xy\in \ell^1$ (see Exercises), which is why it appears all over Analysis.


--

### Hilbert spaces

A _Hilbert space_ $\mathcal H$ is an inner product space with the additional property that all sequences $(x_n)_n$ in $\mathcal H$ whose elements are eventually arbitrarily close to each other (so-called __Cauchy sequences__) __do converge__ to elements in $\mathcal H$. Normed spaces with this property are known as being _complete_. Examples:

* __Hilbert Spaces:__ The inner product spaces from the previous slides ($\mathbb R^d$, $\ell^2$, $L^2$)

* __Non-complete__ inner product spaces: Rational numbers $\mathbb Q$ (equipped with product of numbers), $C([a,b])$ equipped with $\langle\cdot,\cdot \rangle_{L^2([a,b])}$.


<div style="margin-top: 50px;"></div>


<div class="comment">
<b>Note:</b> Any (non-complete) inner product space can be uniquely completed to a Hilbert space by simply including all limits of Cauchy sequences as elements of the space, e.g. the completion of $\mathbb Q$ is $\mathbb R$, the completion of $(C([a,b]), \langle\cdot,\cdot\rangle_{L^2})$ is $L^2([a,b])$. 
</div>


--

### Dual spaces

The (topological) _dual_ $X^\ast$ of a (topological) space $X$ consists of all __continuous linear maps__ (so-called _functionals_) $\phi:X\to \mathbb R$. Examples: 

1. Inner product by a fixed vector $a \in \mathbb R^n$, i.e. $\phi: \mathbb R^n \to \mathbb R$ with $\phi(x) = \langle a,x\rangle_{\mathbb R^n}$.

2. Summation on $\ell^1$ against a fixed bounded sequence $(y_n)_n$ ($\exists C$ s.th. $|y_n|\leq C$ $\forall n$), i.e. $\phi: \mathbb \ell^1 \to \mathbb R$ with $\phi(x)=\sum_n y_n x_n$.

3. Integration on $L^2(\mathbb R)$ against a fixed function $g\in L^2(\mathbb R)$, i.e. $\phi: L^2(\mathbb R) \to \mathbb R$ with $\phi(f) := \int_{\mathbb R} g(x) f(x)\ dx$.

<div style="margin-top: 40px;"></div>

<div class="comment">
<b>Note:</b> 1. and 3. are examples of the general fact that, in any Hilbert space $\mathcal H$, the inner product against a fixed element $y\in\mathcal H$, i.e. $\phi(x) = \langle y,x\rangle_{\mathcal H}$, defines a continuous linear functional $\phi\in \mathcal H^\ast$ (exercise).
</div>


--

### Riesz Representation Theorem

The following theorem shows that the dual $\mathcal H^\ast$ of a Hilbert space $\mathcal H$ can be identified with the Hilbert space itself.

__Theorem__ (Riesz): _For every continuous linear functional $\phi:\mathcal H\to \mathbb R$, there exists a unique element $g_\phi \in \mathcal H$ such that_ 
$$
\phi(f) = \langle g_\phi, f\rangle \quad \forall f\in\mathcal H
$$

Since, the converse is also true (see comment on the previous slide), the mappings $\phi \mapsto g_\phi$ and $g\mapsto \langle g,\cdot \rangle$ are inverses of each other and allow to __identify__ $\mathcal H^\ast$ __with__ $\mathcal H$.


<div style="margin-top: 40px;"></div>

<div class="comment">
<b>Note:</b> For the rigorous identification of $\mathcal H$ and $\mathcal H^\ast$ one also has to think of how the distance measure given by the inner product $\langle\cdot,\cdot\rangle_\mathcal H$ transforms under the bijection (we are not doing this here). 
</div>


--

### Example: Evaluation functionals

For $x\in\mathcal X$, an _evaluation functional_ $\delta_x:\mathcal H \to\mathbb R$ on a Hilbert space $\mathcal H$ of functions $f:\mathcal X\to\mathbb R$ is defined by $$\delta_x(f) := f(x).$$ 

* $\delta_x$ is always linear by definition: $\delta_x(\alpha f+g) = \alpha f(x) + g(x) = \alpha \delta_x(f) + \delta_x(g)$

* $\delta_x$ is not necessarily continuous, e.g. in $L^2(\mathbb R)$, even if $\\|f-f_n\\|\to 0$, the value $f_n(x)$ can be arbitrarily far away from $f(x)$ for any $n\in\mathbb N$ ($\\{x\\}$ has measure $0$).

* In $\mathbb R^d$, evaluation functionals $\delta_i$ map vectors $x$ to single entries $x_i$. Thus,
$$
\delta_i(x) = x_i = \sum\nolimits_{j=1}^d \delta_\{ij\} x_j = \langle (\delta_\{ij\})_\{j=1\}^d, x\rangle
$$
in particular, $\delta_i$ is continuous, and the element $y\in\mathbb R^d$ that is guaranteed to exist by the Riesz representation theorem in this case is $y = (\delta_\{ij\})_\{j=1\}^d$.

--

### Reproducing kernel Hilbert spaces

Let $\mathcal H$ be Hilbert space of functions $\ f:\mathcal X\to \mathbb R$ such that the evaluation functionals $\delta_x:\mathcal H\to \mathbb R$, $f\mapsto f(x)$ are continuous, i.e. $\delta_x \in \mathcal H^\ast$, for all $x\in \mathcal X$. By Riesz' representation theorem, for every $x\in\mathcal X$ there exists an element (i.e. a function) 
\begin{equation}\tag{$\ast$}
k_x\in\mathcal H \quad  \textit{s.th.} \ \ \  f(x) = \langle k_x,f\rangle \quad \forall f\in\mathcal H.
\end{equation}

Any function $K:\mathcal X\times\mathcal X\to\mathbb R$ such that $k_{x'}(x) := K(x, x')$ satisfies ($\ast$) is called a _reproducing kernel for $\mathcal H$_, and $\mathcal H$ is called a _reproducing kernel Hilbert space_ (RKHS) if it has a reproducing kernel (e.g. $\mathbb R^d$, $\ell^2$, not $L^2$).





<div style="margin-top: 40px;"></div>

<div class="comment">
<b>Note:</b> The argument leading to $(\ast)$ shows that any Hilbert space of functions with continuous evaluation functionals ($\delta_x\in\mathcal H^\ast$) is an RKHS. The converse is also true: if $\mathcal H$ has a reproducing kernel $K$, then $\delta_x$ is continuous, since $\delta_x(f) = \langle K(\cdot, y), f\rangle$ and $\langle \cdot,\cdot \rangle$ is continuous in both entries.
</div>




--

### Properties of reproducing kernels

Let $K$ be a reproducing kernel for $\mathcal H$, then

1. $K$ is __unique__ (as a reproducing kernel of $\mathcal H$). <br><span class="comment">Proof: Choose $f=K_1(\cdot, x')-K_2(\cdot,x')$ in $\langle K_1(\cdot,x),f\rangle -\langle K_2(\cdot,x),f\rangle = f(x) - f(x) = 0$.</span>

2. $K$ is an __inner product kernel__: $K(x,x') = \langle K(\cdot,x),K(\cdot,x')\rangle$ <br><span class="comment">Proof: Choose $\ f=k_{x'} = K(\cdot, x')$ in $(\ast)$.</span>

3. $K$ is __symmetric__: $K(x,x') = K(x',x)$ <br><span class="comment">Proof: This directly follows from $2.$ and the symmetry of inner products.</span>

4. $K$ is __positive semi-definite__, i.e. $K_{ij}:=K(x_i,x_j)$ defines a positive semi-definite matrix for any finite set $\\{x_1,\dots,x_n\\}\subset \mathcal X$, i.e. $\sum_\{i,j=1\}^n c_i c_j K_{ij} \geq 0 \ \forall c\in \mathbb R^n$. <br><span class="comment">Proof: $\sum_\{i,j=1\}^n c_i c_j K(x_i,x_j) =  \langle \sum_\{i=1\}^n c_i K(\cdot, x_i), \sum_\{j=1\}^n c_j K(\cdot, x_j)\rangle = \\|\sum_\{i=1\}^n c_i K(\cdot, x_i)\\|^2 \geq 0$</span>




--

### Native spaces

__Theorem__ ([Moore-Aronszajn](https://doi.org/10.2307/1990404)): _A symmetric function $K:\mathcal X\times \mathcal X \to \mathbb R$ is a reproducing kernel for a unique Hilbert space $\mathcal H$ of functions on $\mathcal X$ if $K$ is positive semi-definite._ 


Sketch of proof: 

* Consider the inner product space $V := \mathrm{span} \\{K(\cdot,x): x\in \mathcal X\\}$ of all finite linear combinations $\sum_{i=1}^n\alpha_i K(\cdot, x_i)$ with inner product
$$
\Big\langle \sum\nolimits_i \alpha_i K(\cdot, x_i), \sum\nolimits_j \beta_j K(\cdot, x_j)\Big\rangle_V := \sum\nolimits_\{i,j\} \alpha_i \beta_j K(x_i,x_j)
$$

* Check the reproducing property $(\ast)$: $\ f(x) = \langle K(\cdot, x), f\rangle$ for all $f\in V$.

* Define $\mathcal H$ as the completion of $V$ (the reproducing property still holds).





<div style="margin-top: 40px;"></div>

<div class="comment">
<b>Note:</b> Some books (<a href="https://doi.org/10.1017/CBO9780511617539">Wendland</a>, <a href="https://doi.org/10.1142/6437">Fasshauer</a>) require $K$ to be positive definite in order to get a positive definite inner product, even though semi-definite is enough because one can show $|f(x)|^2 = |\langle K(\cdot, x),f\rangle|^2 \leq K(x,x) \langle f,f\rangle_V $, so $\langle f,f\rangle_V = 0$ implies $\ f=0$ (see e.g. <a href="https://cs.nyu.edu/~mohri/mlbook/">Mohri et al.</a> Sect. 5.2.2, or <a href="https://dl.acm.org/doi/abs/10.5555/648300.755324">Schölkopf et al.</a>, Sect. 1.2).
</div>


--



### Representer theorem

Consider a supervised learning problem for given data $\\{(x_i,y_i)\\}_{i=1}^N \subset \mathcal X\times \mathbb R$. Let $l_f$ be a loss function with respect to a model $f:\mathcal X\to \mathbb R$, e.g. $l_f(x,y) = (y-f(x))^2$. Consider the regularized optimization problem

\begin{equation}\tag{$\ast\ast$}
\min_{f:\mathcal X\to \mathbb R} \ \frac{1}{N}\sum\nolimits_{i=1}^N l_f(x_i,y_i) + \lambda \ g(\\|f\\|)
\end{equation}

where $g:\mathbb R_+\to\mathbb R$ is a strictly monotonically increasing function, e.g. $g(t) = t^2$, and $\\|f\\|$ is some a function norm. 

__Theorem:__ _If the minimization in $(\ast\ast)$ is restricted to an RKHS $\mathcal H$ with kernel $K$ and $\\|\cdot\\| = \sqrt{\langle \cdot,\cdot\rangle_\{\mathcal H\}}$, then each minimizer of $(\ast\ast)$ admits a representation of the form_
$$
f(x) = \sum\nolimits_{i=1}^N \alpha_i \ K(x_i, x)
$$
_where $\alpha = (\alpha_1,\dots,\alpha_N)\in\mathbb R^N$ is the only degree of freedom that is left._




---


## 3. Kernel machines


--


### Linear Support Vector Machine

Consider a binary classification problem for a dataset $\\{(x_i,y_i)\\}_{i=1}^N$, $y_i\in \\{-1,1\\}$.

* _Parametrized hyperplane_ $\ h_{w,b} := \\{\xi| \langle w,\xi\rangle + b = 0\\}$ 

* __Decision function__ $\ f_{w,b}(x) := \mathrm{sgn} (\langle w,x\rangle + b) \in \\{-1,1\\}$ 

* _Margin_ $m_{w,b} :=$ __distance of__ $h_{w,b}$ __to closest points__  $=$ $\pm\big(\big\langle \frac{w}{\\|w\\|}, x_\{\pm\}^\ast \big\rangle + \frac{b}{\\|w\\|} ) $ 

* _Scaling invariance:_ $h_{w,b} = h_\{\alpha w,\alpha b\}$ and $m_\{w,b\} = m_\{\alpha w,\alpha b\}$ for any $\alpha\not=0$.

* _Scaling trick (canonical form):_ Rescale $w$ such that $\\|w\\| = \frac{1}{m_{w,b}}$ (dep. on $w$ and $b$), resulting in 
$\langle w,x_\pm^\ast\rangle + b = \pm 1$ and $m_{w,b} = \frac{1}{\\|w\\|}$.

* _Max. margin classifier_ (__linear SVM__): $\ \min_{w,b} \frac{1}{2}\\|w\\|^2$ s.t. $y_i(\langle w,x_i\rangle + b) \geq 1 \ \forall i$.

See [notes](pdfs/LS2_InstanceBasedLearning_LinearSVM.pdf) for details.


--

### Transforming constrained to unconstrained optimization

A __constrained__ optimization problem 
\begin{equation}\tag{$\ast$}
\min\nolimits_\omega f(\omega) \quad \text{subject to} \ c_i(\omega)\leq 0 \ \forall i\in\{1,\dots,N\}
\end{equation}
can be formally translated to the __unconstrained__ problem $\inf_\omega F(\omega)$ where 
$$
F(\omega) = \left\\{\begin{array}{cc} f(\omega) & \text{if } c_i(\omega)\leq 0 \ \forall i\in\\{1,\dots, N\\} \\\\ \infty & \text{otherwise}\end{array}\right.
$$

<div style="margin-top: 50px;"></div>

_Main example:_ $F(\omega) = \sup_\{\lambda_i\geq 0\} \mathcal L(\omega,\lambda)$ with the __Lagrangian__ $$\mathcal L(\omega,\lambda) := f(\omega) + \sum_{i=1}^N\lambda_i c_i(\omega),$$
so that $(\ast)$ can be written as
$\inf_\omega\sup_\{\lambda_i\geq 0\} \mathcal L(\omega, \lambda)$.


--



### Duality in constrained optimization

So far (trivial): 
$ \ \ \min\nolimits_\omega f(\omega) \ \text{s.t.} \ c_i(\omega)\leq 0 \ \forall i\in\{1,\dots,N\}
\ \ \Longleftrightarrow \ \ \inf_\omega\sup_\{\lambda_i\geq 0\} \mathcal L(\omega, \lambda)
$

_Strong duality:_ Can we __interchange the $\sup$ and $\inf$ operators__? <br>
More precisely, strong duality means that, if $g(\lambda):=\inf_\{\omega\} \mathcal L(\omega,\lambda)$, then
$$
\underbrace\{\inf_\omega F(\omega)\}_\{\textit{Primal Problem}\}  = \underbrace\{\sup_\{\lambda_i\geq 0\} g(\lambda)\}_\{\textit{Dual Problem}\}
$$

<div class="fragment">

_Examples of sufficient conditions for strong duality_ (there are many!):
* $f$ and all $c_i$ are affine functions (linear optimization problem)
* $f$ is __convex and all $c_i$ are affine__ (variant of _Slater's condition_) 
* $f$ and all $c_i$ are convex and continuous on a compact and convex domain (_minimax thm._)     

</div>

<div class="fragment">

__Theorem__ ([Bazaraa et al. 2006](https://doi.org/10.1002/0471787779), Thm. 6.2.5): _$\omega^\ast$ and $\lambda^\ast$ are solutions of the primal and dual problems, respectively, and strong duality holds, if and only if $(\omega^\ast,\lambda^\ast)$ is a saddle point of $\mathcal L$, i.e. $\mathcal L(\omega^\ast,\lambda) \leq \mathcal L(\omega^\ast,\lambda^\ast) \leq \mathcal L(\omega,\lambda^\ast)$ for all $\omega, \lambda$._

</div>

--


### Karush-Kuhn-Tucker (KKT) conditions

Assume that strong duality holds for a pair $\omega^\ast, \lambda^\ast$ (and that $f$, $c_i$ are differentiable), then 

* $c_i(\omega^\ast) \leq 0$ and $\lambda^\ast_i \geq 0$ for all $i=1,\dots, N$ (_feasability_)

* $\frac{\partial \mathcal L}{\partial \omega}(\omega^\ast,\lambda^\ast) = 0$ (_stationarity_ of $\mathcal L(\omega,\lambda^\ast)$ at $\omega = \omega^\ast$) 

* $\lambda^\ast_i c_i(\omega^\ast) = 0$ for all $i=1,\dots,N$ (_complementary slackness_)

These are known as the _Karush-Kuhn-Tucker (KKT) conditions_.  


__Theorem__ (see e.g. [Chi et al. 2017](https://doi.org/10.1201/9781315366920), Sect. 9.5): <br>
$(i)$ _If strong duality holds, then the above conditions follow for a pair $\omega^\ast,\lambda^\ast$ of solutions._<br>
$(ii)$ _For convex problems with strong duality (e.g. Slater's condition holds), the KKT conditions are also sufficient for $\omega^\ast$, $\lambda^\ast$ being solutions for the primal and dual problems, respectively._




<div style="margin-top: 40px;"></div>

<div class="comment">
<b>Note:</b> One can find many regularity conditions in the optimization literature (so-called <i>constraint qualifications</i>, e.g. <a href="https://www.jstor.org/stable/2028581">Peterson, 1973</a>) under which the KKT conditions are necessary, but one does not necessarily have strong duality.
</b> 
</div>


--

### Dual problem for linear SVM

* __Primal problem:__ $\ \min_{w,b} \frac{1}{2}\\|w\\|^2$ subject to $1-y_i(\langle w,x_i\rangle + b) \leq 0 \ \forall i \in \\{1,\dots,N\\}$ 

* __Lagrangian:__ $\ \mathcal L(w,b, \lambda) =  \frac{1}{2}\\|w\\|^2 + \sum_{i=1}^N \lambda_i \big(1-y_i(\langle w,x_i\rangle + b) \big)$

* Since $f(w) = \frac{1}{2}\\|w\\|^2$ is convex, and the constraints are affine (_variant of Slater's condition_), we have __strong duality__. In particular, the KKT conditions are necessary and sufficient. Moreover, we can maximize $g(\lambda) := \min_{w,b} \mathcal L(w,b,\lambda) = \mathcal L(w^\ast(\lambda),b^\ast(\lambda),\lambda)$ over $\lambda_i\geq 0$, where $w^\ast(\lambda)$ and $b^\ast(\lambda)$ satisfy
$$
\underbrace\{\frac{\partial \mathcal L}{\partial w_i}(w^\ast(\lambda),b^\ast(\lambda),\lambda) = 0\}_\{w^\ast(\lambda) = \sum_\{i=1\}^N \lambda_i y_i x_i\} \ , \quad \underbrace\{\frac{\partial \mathcal L}{\partial b}(w^\ast(\lambda), b^\ast(\lambda), \lambda) = 0\}_\{\sum_\{i=1\}^N \lambda_i y_i = 0\} 
$$

$\Rightarrow$ __Dual problem__ (see [notes](pdfs/LS2_InstanceBasedLearning_LinearSVM-DualProblem.pdf) for details): $\  \max_{\lambda_i\geq 0} g(\lambda)$ subject to $\sum_{i=1}^N \lambda_i y_i = 0$, where
$$
g(\lambda) =  \mathcal L(w^\ast(\lambda),b^\ast(\lambda),\lambda) = \sum\nolimits_{i=1}^N \lambda_i - \tfrac{1}{2} \sum\nolimits_{i,j=1}^N \lambda_i \lambda_j y_i y_j \langle x_i, x_j\rangle
$$




--

### Support Vectors

By complementary slackness, $\lambda_i^\ast(1-y_i(\langle w^\ast,x_i\rangle + b^\ast)) = 0$ for $i=1,\dots,N$, i.e.  
\begin{equation}\tag{$\ast$}
\lambda_i^\ast = 0 \quad \text{or} \quad y_i(\langle w^\ast,x_i\rangle + b^\ast) = 1
\end{equation}
This means that in the linear combination $w^\ast(\lambda^\ast) = \sum_i \lambda_i^\ast y_i x_i$ only those patterns $x_i$ contribute that satisfy the constraint as an equality (they are on the margin!) known as _support vectors_.  In particular, __all other patterns have no influence__ on the optimal hyperplane.

<div style="margin-top: 40px;"></div>

### Decision function

Plugging in the expression for $w^\ast(\lambda^\ast)$ into the decision function $f_{w,b}$ of the linear SVM, we obtain
$$
f_{w^\ast(\lambda^\ast),b^\ast}(x) = \mathrm{sgn} \Big(\sum\nolimits_{i=1}^N \lambda_i^\ast y_i \langle x_i,x\rangle + b^\ast \Big)
$$
where $\lambda^\ast$ is given by the dual problem, and, due to $(\ast)$, $b^\ast = y_j - \sum_{i=1}^N \lambda_i^\ast y_i \langle x_i,x_j\rangle$ for all $j$ with $\lambda_j^\ast > 0$ (e.g. by averaging).


--

### Nonlinear SVM

In the linear SVM with decision function $x\mapsto \mathrm{sgn} (\sum\nolimits_{i=1}^N \lambda_i^\ast y_i \langle x_i,x\rangle + b^\ast)$, a new pattern $x$ is __compared with all support vectors__ $x_i$ using $\langle x_i,x\rangle$ as similarity measure and then categorized based on the weighted sum of these similarities.

Above, the dimension of the $x_i$ was arbitrary. Thus we can __replace them by their image under a feature map__ $\phi:\mathcal X\to\mathcal H$ into a feature space $\mathcal H$ with inner product $\langle \cdot,\cdot \rangle_\{\mathcal H\}$, so that 
$$
f(x) =\mathrm{sgn} \Big(\sum\nolimits_{i=1}^N \lambda_i^\ast y_i K(x_i,x) + b^\ast \Big)
$$
where $K(x,x'):=\langle \phi(x),\phi(x') \rangle_{\mathcal H}$ defines an inner product kernel, and $\lambda^\ast$ is a solution to the dual problem $\max_{\lambda_i\geq 0} g(\lambda)$ s.t. $\sum_{i=1}^N \lambda_i y_i = 0$, where in, analogy to the linear SVM,
$$
g(\lambda) = \sum\nolimits_{i=1}^N \lambda_i - \tfrac{1}{2} \sum\nolimits_{i,j=1}^N \lambda_i \lambda_j y_i y_j K(x_i, x_j)
$$


<div style="margin-top: 30px;"></div>

<div class="comment">
<b>Note:</b> In exercises for this section you will use these results to create simulations for linear and nonlinear Support Vector Machines. You can view my implementations <a target="_blank" href="https://share.streamlit.io/sgttwld/learningsystems2/InstanceBasedLearning/streamlit_app.py">here</a>.
</b> 
</div>


--

### Extensions of standard SVMs

* _Soft Margin_ (in case of overlapping classes due to noisy data): Introduce _slack variables_ $\xi_i \geq 0$, relax the constraints to $y_i(\langle w,x_i\rangle + b) \geq 1-\xi_i$, and minimize $C \sum_i \xi_i$ additionally to $\\|w\\|^2$, where $C>0$ denotes a trade-off parameter. The corresponding dual problem takes exactly the same form as the hard margin SVM from the previous slides, with the additional constraint that $\lambda_i \leq C$ (so that $\lambda_i\in [0,C]$) $\forall i=1,\dots,N$. 

* _SVM Regression (linear and kernel regression):_ Analogous to soft margins, one introduces _slack variables_ $\xi_i, \xi_i^\ast \geq 0$ and minimizes $\\|w\\|^2 + C \sum_i (\xi_i + \xi_i^\ast)$ subject to $f(x_i)-y_i \leq \varepsilon + \xi_i$ and $y_i-f(x_i)\leq \varepsilon +\xi_i^\ast$ for some $\epsilon > 0$, where $f(x) := \langle w, x\rangle + b$. This can be transformed to a dual problem with Lagrange multipliers $\lambda_i,\lambda_i^\ast$ and a decision function of the form $f(x) = \sum_i (\lambda_i^\ast-\lambda_i)K(x_i,x) +b$.  


--

### The kernel trick

Consider a learning algorithm whose prediction function takes the form 
$$
f(x) = F\big( \sum\nolimits_{i=1}^N \alpha_i(y_i) K(x_i,x) + b \big)
$$
where $K$ is some inner product kernel. Then we can obtain a new algorithm by simply replacing the kernel $K$ by another inner product kernel $K'$.


Examples include

* Linear SVM (classification) $\Longrightarrow$ __Nonlinear/kernel SVM__ 

* Linear SVM Regression $\Longrightarrow$ __Kernel regression__

* Principal component analysis (PCA) $\Longrightarrow$ __Kernel PCA__


--

### Instance-based methods not relying on inner products

* _$k$ nearest neighbour_ classification: Choose the label that is most common under the $k$ nearest neighbours (given some notion of distance).

* _k nearest neighbour_ regression: Average the values of the $k$ nearest neighbours. 

* _RBF network_ regression: $f(x) = \sum_n \alpha_n h(\\|x-x_n\\|)$ for some localized function $h$ (usually a Gaussian)


