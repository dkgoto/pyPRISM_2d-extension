This repository extends the Python-based open-source framework [pyPRISM](https://github.com/usnistgov/pyPRISM) to two spatial dimensions, whereas the original implementation is designed for three dimensions.

"Domain_extended.py" is specifically designed to extend the pyPRISM framework to 2D. This file should replace "pyPRISM/core/Domain.py" in the original pyPRISM directory.

When defining the system domain in 2D, an additional argument, dim, must be set to 2 (default: 3), as shown below:

```sh
sys.domain = pyPRISM.Domain(dr=0.01, length=4096, dim=2)
```

### Hankel Transform
In two dimensions, the Fourier transform of radially symmetric functions simplifies to the Hankel transform, which can be efficiently computed using the quasi-discrete Hankel transform (QDHT). This functionality is implemented via the [PyHank](https://github.com/etfrogers/pyhank) library.

### About pyPRISM
pyPRISM solves the molecular Ornstein-Zernike equation given a closure approximation and a pair potential. It enables the computation of various structural and thermodynamic properties in equilibrium homogeneous liquid states.
See [pyPRISM](https://github.com/usnistgov/pyPRISM) for more details, and follow the instructions therein for citations.


### Last Update
Oct 19, 2022.
