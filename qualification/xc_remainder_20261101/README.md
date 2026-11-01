# Beyond-quadratic semilocal-XC diagnostic

Exact frozen-orbital XC remainders are evaluated beyond the complete second-order
spin-density expansion, including kinetic-energy density for M06-2X. The tests
reuse Oracle quadrature caches and do not run a new DFT SCF. They isolate DFT
nonlinearity from the preceding native-GFN2-only remainder diagnostic.

The corrections are small for the problematic pi-pi states and do not provide
a general repair. These are frozen-orbital diagnostic values, not stationary
predictions or a new production semiempirical XC implementation.
