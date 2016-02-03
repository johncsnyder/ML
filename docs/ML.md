# ML


---

<a id="method__center.1" class="lexicon_definition"></a>
#### center(K) [¶](#method__center.1)
centering kernel matrices

$\varphi = \frac{1}{n} \sum_i \varphi(x_i)$

$\tilde K_{ij} = (\varphi(x_i) - \tilde\varphi)^T (\varphi(x_j) - \tilde\varphi)$

$\tilde K_{ij} = K_{ij} - \frac{1}{n} \sum_j K_{ij} - \frac{1}{n} \sum_i K_{ji} + \frac{1}{n^2} \sum_{ij} K_{ij}$

$\tilde L_{ij} = L_{ij} - \frac{1}{n} \sum_j K_{ij} - \frac{1}{n} \sum_i L_{ji} + \frac{1}{n^2} \sum_{ij} K_{ij}$

$\tilde M_{ij} = M_{ij} - \frac{1}{n} \sum_j L_{ij} - \frac{1}{n} \sum_i L_{ji} + \frac{1}{n^2} \sum_{ij} M_{ij}$



*source:*
[ML/src/ML.jl:710](https://github.com/johncsnyder/ML.jl/tree/df56ab5ddd2a6746d92affddefe064dd71c97de2/src/ML.jl#L710)

