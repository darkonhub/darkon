"""Copyright 2017 Neosapience, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from __future__ import unicode_literals

import abc


class InfluenceFeeder:
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """ reset dataset
        """
        pass

    @abc.abstractmethod
    def train_batch(self, batch_size):
        """ train_batch

        Parameters
        ----------
        batch_size :

        Returns
        -------
        xs : feed input values
        ys : feed label values
        """
        pass

    @abc.abstractmethod
    def train_one(self, index):
        """ train_one

        Parameters
        ----------
        index :

        Returns
        -------
        x : feed one input value
        y : feed one label values
        """
        pass

    @abc.abstractmethod
    def test_indices(self, indices):
        """ test_indices

        Parameters
        ----------
        indices :

        Returns
        -------
        x : feed input values
        y : feed label values
        """
        pass
