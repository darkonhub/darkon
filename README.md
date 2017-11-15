**Darkon: Performance hacking for your deep learning models**

**Darkon** is an open source toolkit for improving and debugging deep learning models.
People think that deep neural network is a black-box that requires only large dataset and expect learning algorithms returns well-performing models. 
However, trained models often fail in real world usages, and it is difficult to fix such failure due to the black-box nature of deep neural networks.
We are developing **Darkon** to ease effort to improve performance of deep learning models. 

In this first release, we provide influence score calculation easily applicable to existing Tensorflow models (other models to be supported later)
Influence score can be used for filtering bad training samples that affects test performance negatively. 
It can be used for prioritize potential mislabeled examples to be fixed, and debugging distribution mismatch between train and test samples.

**Darkon** will gradually provide performance hacking methods easily applicable to existing projects based on following technologies.
- Dataset inspection/filtering/management
- Continual learning
- Meta/transfer learning
- Interpretable ML
- Hyper parameter optimization
- Network architecture search

More features will be released soon. Feedback and feature request are always welcome, which help us to manage priorities. Please keep your eyes on **Darkon**. 

## Dependencies

[![Join the chat at https://gitter.im/darkonhub/darkon](https://badges.gitter.im/darkonhub/darkon.svg)](https://gitter.im/darkonhub/darkon?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
- [Tensorflow](https://github.com/tensorflow/tensorflow)>=1.3.0

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
**Apache License 2.0**

## References
[1] Pang Wei Koh and Percy Liang "[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)" ICML2017
