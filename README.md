# The logistic function of every element over CKKS — DESILO

> Implements [`logistic` / `f64@1.0.0`](https://www.fherma.io/kernels/logistic/specifications/f64)
> on the FHERMA kernel catalogue, with [DESILO FHE](https://fhe.desilo.dev/).

```text
kernel logistic<N: u32>(
    %xs: secret<tensor<N x f64>>,
) -> %y: secret<tensor<N x f64>>
```

`1 / (1 + e^-x)` over [-25, 25], held within 1e-3 everywhere.

## The polynomial here is ours

This is the one place these three DESILO answers differ from the OpenFHE ones
beside them, and the board should be read with that in mind.

The published LogisticFunction component of
[polycircuit](https://github.com/fairmath/polycircuit) is a Chebyshev series
over [-25, 25] whose high-degree terms are unrolled by hand in several groups
of different shape. Reproducing that circuit term by term is work out of
proportion to what it would buy, so `init` computes a plain Chebyshev
projection of the logistic function instead, at degree 59 — the degree the
published component uses.

The coefficients are therefore computed from the function rather than carried
as a table, and no number in this repository is a fit.

## The engine takes the machine

`mode` is `parallel`, not `cpu`. They are different engines, not two speeds of
one: `cpu` computes in a single thread, and a number measured there is a number
about one core, which is not what the answers beside this one are measured at.
The thread count is the cores the runner reports rather than the library's own
default of four, for the same reason.

## Accuracy

The projection reaches 3.4e-4 on the specification's domain, inside the 1e-3
bar by a factor of three, and the measured answer reaches the same: the scheme's
noise is below the polynomial's own error here.

The Chebyshev basis is on [-1, 1] and the data is on [-25, 25], so the first
operation of the circuit is the scaling, which costs one level. The level
budget is 8, which is what the circuit spends.

## Running it yourself

In the `fherma/desilo:1.17.0` image, or anywhere the wheel installs:

```sh
pip install --no-cache-dir --target . desilofhe==1.17.0
python main.py <point directory>
```

A point directory is what the specification's testing bundle writes with
`main.py make`; the same bundle judges the result with `main.py verify`.

## Layout

```
solution/
  solve.py       the four functions — the only file written by hand
  config.jsonc   the engine: scheme, mode, level budget, which keys
solution-gpu/    the same four functions, with the engine on a card
  fherma.toml    what it implements, and with what
  envelope.py    generated — engine, keys, encryption. Holds the secret key
  main.py        generated — the measured loop
  fherma.py      generated — the types, from the signature
```

Everything but `solve.py` and `config.jsonc` is emitted by `fherma-lang` from the
specification's signature and replaced at every measurement, so a solution
cannot drift from the contract it claims to meet.

## Licence

The solution is Apache-2.0. The DESILO library is not redistributed here: the
build installs it from PyPI, under its own licence, which permits
non-commercial use.
