# Implementation of the `Adpredictor` algorithm: A bayesian click-through rate prediction Algorithm for advertising

This repo reproduces the algorithm described in the paper:
[Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft's Bing Search Engine](https://www.microsoft.com/en-us/research/wp-content/uploads/2010/06/AdPredictor-ICML-2010-final.pdf) by the the legendary team at Microsoft Research, including `Thore Graepel`, `Joaquin Quiñonero Candela`, `Thomas Borchert`, and `Ralf Herbrich`.

> This particular implementation is based on Andrew Tulloch's [blog post](http://tullo.ch/articles/online-learning-with-adpredictor/) and companion code. The code found here has been modernized and adapted to work with Python 3, and is intended to be used as a learning resource for those interested in understanding bayesian algorithms.

# How to run

After cloning the repository, you can call `make demo` to run the code. The output will be saved in the `logs/adpredictor/` directory.