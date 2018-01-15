<div align="center">
    <img src="https://cdn.rawgit.com/darkonhub/darkon/d026f574/brand/logo.png" width="400"><br><br>
</div>

[![Build Status](https://travis-ci.org/darkonhub/darkon.svg?branch=master)](https://travis-ci.org/darkonhub/darkon)
[![codecov](https://codecov.io/gh/darkonhub/darkon/branch/master/graph/badge.svg)](https://codecov.io/gh/darkonhub/darkon)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/darkon.svg?style=flat-square)](https://pypi.python.org/pypi/darkon)
[![Gitter](https://badges.gitter.im/darkonhub/darkon.svg)](https://gitter.im/darkonhub/darkon?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/077f07f7a52b4d8186beee724ed19231)](https://www.codacy.com/app/zironycho/darkon?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=darkonhub/darkon&amp;utm_campaign=Badge_Grade)

---------------------------------------------------

**Darkon: Toolkit to Hack Your Deep Learning Models**

**Darkon** is an open source toolkit to understand deep learning models better. Deep learning is often referred as a black-box that is difficult to understand.
But, accountability and controllability could be critical to commercialize deep learning models. People often think that high accuracy on prepared dataset 
is enough to use the model for commercial products. However, well-performing models on prepared dataset often fail in real world usages and cause corner cases 
to be fixed. Moreover, it is necessary to explain the result to trust the system in some applications such as medical diagnosis, financial decisions, etc. We hope  
**Darkon** can help you to understand the trained models, which could be used to debug failures, interpret decisions, and so on. 

Here, we provide functions to analyze deep learning model decisions easily applicable to any Tensorflow models (other models to be supported later).
Influence score can be useful to understand the model through training samples. The score can be used for filtering bad training samples that affects test performance negatively. 
It is useful to prioritize potential mislabeled examples to be fixed, and debug distribution mismatch between train and test samples.
In this version, we have added Grad-CAM and Guided Grad-CAM, which are useful to understand decisions of CNN models. 

We will gradually enable technologies to analyze deep learning models easily applicable to your existing projects.
More features will be released soon. Feedback and feature request are always welcome, which help us to manage priorities. Please keep your eyes on **Darkon**. 

## Dependencies
- [Tensorflow](https://github.com/tensorflow/tensorflow)>=1.3.0

## Installation
Install Darkon alone
```bash
pip install darkon
```
Install with TensorFlow CPU
```bash
pip install darkon[tensorflow]
```
Install with TensorFlow GPU
```bash
pip install darkon[tensorflow-gpu]
```

## Examples 
- [Examples](https://github.com/darkonhub/darkon-examples) 

## API Documentation
- [Documentation](http://darkon.io/api)

## Communication
- [Issues](https://github.com/darkonhub/darkon/issues): report issues, bugs, and request new features
- [Pull request](https://github.com/darkonhub/darkon/pulls)
- Discuss: [Gitter](https://gitter.im/darkonhub/darkon?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
- Email: [contact@darkon.io](mailto:contact@darkon.io) 

## Authors
[Neosapience, Inc.](http://www.neosapience.com)

## License
**Apache License 2.0**

## References
[1] Cook, R. D. and Weisberg, S. "[Residuals and influence in regression](https://www.casact.org/pubs/proceed/proceed94/94123.pdf)", New York: Chapman and Hall, 1982

[2] Koh, P. W. and Liang, P. "[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)" ICML2017

[3] Pearlmutter, B. A. "[Fast exact multiplication by the hessian](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf)" Neural Computation, 1994

[4] Agarwal, N., Bullins, B., and Hazan, E. "[Second order stochastic optimization in linear time](https://arxiv.org/abs/1602.03943)" arXiv preprint arXiv:1602.03943

[5] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra "[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)" ICCV2017

