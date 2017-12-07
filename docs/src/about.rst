About Darkon
============



.. image:: https://travis-ci.org/darkonhub/darkon.svg?branch=master
   :target: https://travis-ci.org/darkonhub/darkon
   :alt: Build Status


.. image:: https://codecov.io/gh/darkonhub/darkon/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/darkonhub/darkon
   :alt: codecov


.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License


.. image:: https://img.shields.io/pypi/v/darkon.svg?style=flat-square
   :target: https://pypi.python.org/pypi/darkon
   :alt: PyPI


.. image:: https://badges.gitter.im/darkonhub/darkon.svg
   :target: https://gitter.im/darkonhub/darkon?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
   :alt: Gitter


.. image:: https://api.codacy.com/project/badge/Grade/077f07f7a52b4d8186beee724ed19231
   :target: https://www.codacy.com/app/zironycho/darkon?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=darkonhub/darkon&amp;utm_campaign=Badge_Grade
   :alt: Codacy Badge


----

**Darkon: Performance hacking for your deep learning models**

**Darkon** is an open source toolkit for improving and debugging deep learning models.
People think that deep neural network is a black-box that requires only large dataset and expect learning algorithms returns well-performing models.
However, trained models often fail in real world usages, and it is difficult to fix such failure due to the black-box nature of deep neural networks.
We are developing **Darkon** to ease effort to improve performance of deep learning models.

In this first release, we provide influence score calculation easily applicable to existing Tensorflow models (other models to be supported later)
Influence score can be used for filtering bad training samples that affects test performance negatively.
It can be used for prioritize potential mislabeled examples to be fixed, and debugging distribution mismatch between train and test samples.

**Darkon** will gradually provide performance hacking methods easily applicable to existing projects based on following technologies.


* Dataset inspection/filtering/management
* Continual learning
* Meta/transfer learning
* Interpretable ML
* Hyper parameter optimization
* Network architecture search

More features will be released soon. Feedback and feature request are always welcome, which help us to manage priorities. Please keep your eyes on **Darkon**.

Dependencies
------------


* `Tensorflow <https://github.com/tensorflow/tensorflow>`_\ >=1.3.0

Installation
------------

.. code-block:: bash

   pip install darkon

Usage
-----

.. code-block:: python

   inspector = darkon.Influence(workspace_path,
                                YourDataFeeder(),
                                loss_op_train,
                                loss_op_test,
                                x_placeholder,
                                y_placeholder)

   scores = inspector.upweighting_influence_batch(sess,
                                                  test_indices,
                                                  test_batch_size,
                                                  approx_params,
                                                  train_batch_size,
                                                  train_iterations)

Communication
-------------


* `Issues <https://github.com/darkonhub/darkon/issues>`_\ : report issues, bugs, and request new features
* `Pull request <https://github.com/darkonhub/darkon/pulls>`_
* Discuss: `Gitter <https://gitter.im/darkonhub/darkon?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>`_
* Email: `darkon@neosapience.com <mailto:darkon@neosapience.com>`_

Authors
-------

`Neosapience, Inc. <http://www.neosapience.com>`_

License
-------

**Apache License 2.0**

References
----------

[1] Pang Wei Koh and Percy Liang "\ `Understanding Black-box Predictions via Influence Functions <https://arxiv.org/abs/1703.04730>`_\ " ICML2017

