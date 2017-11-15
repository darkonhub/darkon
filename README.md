[![Build Status](https://travis-ci.org/darkonhub/darkon.svg?branch=master)](https://travis-ci.org/darkonhub/darkon)
---------------------------------------------------

**Darkon: Performance hacking for your AI model**

**Darkon** is an open source software library for improving and debugging deep learning models. People think that deep neural network is a black-box that requires only large dataset and expect learning algorithms returns well-performing models. However, trained models often fail in real world usages, and it is difficult to fix such failure due to the black-box nature of deep neural networks.

**Darkon** will gradually provide performance hacking features easily applicable to your existing projects based on following technologies.
- Dataset inspection/filtering
- Continual learning
- Meta/transfer learning
- Hyper parameter optimization
- Network architecture search

In this first release, we provide influence function calculation feature easily applicable to any Tensorflow models. Influence score can be used for filtering bad training samples that affects test performance negatively. It can be used for prioritize potential mislabeled examples to be annotated, and debugging distribution mismatch between train and test samples.

More features will be released soon. Please keep your eyes on **Darkon**

## Dependencies
- [Tensorflow>=1.3.0](https://github.com/tensorflow/tensorflow)

## Installation
```bash
pip install darkon
```

## Examples / Getting Started 
- [Examples](https://github.com/darkonhub/darkon-examples) 
- Documentation: ~~will be soon~~  

## Communication
- [Issues](https://github.com/darkonhub/darkon/issues): report issues, bugs, and request new features
- [Pull request](https://github.com/darkonhub/darkon/pulls)
- News: ~~link twitter account~~
- Discuss: ~~gitter~~
- Email: [darkon@neosapience.com](mailto:darkon@neosapience.com) 

## Authors
[Neosapience Inc.](http://www.neosapience.com)

## License
**Apache License v2.0**

## References
[1] Pang Wei Koh and Percy Liang "[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)" ICML2017]
