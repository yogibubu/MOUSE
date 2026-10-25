# MOUSE

**Molecular Orbital Utility for Semiempirical Explorations**

MOUSE is a tight-binding platform for molecular geometry optimization,
ground- and excited-state derivatives, and subtractive QM/QM′ models.
It combines FAST-XTB and FAST-TDA with ORACLE molecular perception,
SMITH coordinates, the LINK optimizer and APOC electronic-state tracking.

This standalone distribution includes these components and their complete
mandatory internal Python dependencies. Installation needs no private
repository and no MATRIX checkout. The original `matrix_*` Python module
names are retained so that published calculation scripts remain usable.

## Install from source

Use Python 3.11–3.13 and a C++17 compiler. macOS uses the system Accelerate
framework. Linux needs development headers/libraries for BLAS and LAPACK
(or OpenBLAS). On Debian/Ubuntu, install `build-essential libblas-dev
liblapack-dev python3-dev python3-venv` before building.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python install.py
python examples/ground_state.py
python examples/excited_state.py
python examples/geometry.py
```

The installer builds and installs all bundled internal packages together,
then resolves the public NumPy/SciPy and other Python dependencies, including the PyTorch
runtime used by the analytic ALPB response. No
external electronic-structure executable is required for native GFN2-xTB
or TDA-xTB calculations. Adapters are included; their optional external
programs and licenses must be supplied separately when those adapters are used.

## Install a prebuilt release

Download the release archive matching your Python version and operating
system, extract it and activate a fresh virtual environment. Then run:

```bash
python install.py --offline
```

The platform archive includes both the internal wheels and external Python
dependency wheels. The supplied macOS ARM64/CPython 3.13 archive requires
macOS 14 or newer (the bundled ALPB numerical runtime requirement).
A macOS ARM64 archive cannot be used on Linux, Intel
macOS or another CPython version. The source distribution is portable to
the supported systems with the build prerequisites above.

## Capabilities and scope

- Resident GFN2-xTB ground-state energies, gradients and analytic Hessians.
- TDA-xTB and spin-flip response, including analytic derivative operators.
- Ground-reference ALPB solvent contributions in the same derivative framework.
- Sparse coordinate operations and bounded-memory optimization through LINK.
- State identity and transition-density overlap through APOC.
- Subtractive QM/QM′ energy, gradient and Hessian assembly with explicit caps.
- Immutable optional parameter profiles, including the one-scale,
  oxygen-centered water example described in the accompanying work.

The compact TDA example illustrates the derivative API. Its raw native
energy gaps are model dependent and are not an optical accuracy benchmark;
QM/QM′ replaces the low-level chromophore energy by a high-level term.

The water profile is an example of parameter development with analytic
derivatives. Standard GFN2 remains the default; extensive parametrization
and general water interaction energies are outside that calibration example.
The near-linear timings reported for long chains concern the coordinate
path. Dense molecular electronic response has a separate size dependence.

The complete optimization and vibrational applications in the manuscript
reach 149 atoms. The chlorophyll infrared example demonstrates feasibility.
The hydrated CT controls concern intramolecular chromophore excitations.

## Connected exploration and spectroscopy tools

SEEKER, TRINITY and SATI are separate packages and are not included in this
standalone distribution. The MOUSE examples cover electronic derivatives
and geometry optimization. Reproducing the connected vibrational and vibronic
workflows also requires TRINITY/SATI.

The accompanying manuscript cites the exploration method in
[Digital Discovery 2022](https://doi.org/10.1039/D2DD00070A) and the
[TRINITY framework](https://github.com/yogibubu/TRINITY-coupled-oscillators).

## Build a release locally

```bash
python -m pip install build setuptools==84.0.0 wheel==0.48.0 numpy==2.3.5
python tools/build_wheels.py --download-dependencies
```

Builds run locally and consume no GitHub Actions quota. Keep the generated
wheels in `dist/wheels` for installation; do not mix native wheels built
for different operating systems or Python versions in an offline archive.

## Reproducibility and citation

`SOURCE_RECORD.json` records the selected package source revision and a
SHA-256 digest for every original source file. `RELEASE_RECORD.json`
records the distributable files, platform and installation/example results.
`CITATION.cff` supplies software citation metadata. The manuscript and
Supporting Information describe the scientific models and numerical checks.

Repository: https://github.com/yogibubu/MOUSE

Releases: https://github.com/yogibubu/MOUSE/releases

## License

Original code is distributed under the BSD 3-Clause license in `LICENSE`.
Third-party software retains its own license. External dependency wheels
include the corresponding license metadata; they are not relicensed as MOUSE.
