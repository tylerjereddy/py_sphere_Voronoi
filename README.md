Voronoi diagrams on the surface of a Sphere
===========================================

A Python module for obtaining Voronoi diagrams on the surfaces of spheres, including the calculation of Voronoi region surface areas. Applications may range from calculating area per lipid in spherical viruses to geographical parsing.

The documentation for the project is available here: http://py-sphere-voronoi.readthedocs.org/en/latest/voronoi_utility.html

This project is still in development and some of the algorithm weaknesses are highlighted in the above documentation.

Please cite: [![DOI](https://zenodo.org/badge/6247/tylerjereddy/py_sphere_Voronoi.svg)](http://dx.doi.org/10.5281/zenodo.13688)

For contributions:
  * ensure most/all unit tests pass (run: nosetests)
  * ensure all doctests pass (run: python voronoi_utility.py)
  * if you import new modules, you may need to mock them in the Sphinx conf.py documentation file so that the docs are properly compiled by readthedocs
  * attempt to match the [numpy documentation standard](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) as closely as possible
