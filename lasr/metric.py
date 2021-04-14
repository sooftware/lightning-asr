# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import Levenshtein as Lev


class ErrorRate(object):
    """
    Provides inteface of error rate calcuation.
    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, vocab) -> None:
        self.total_dist = 0.0
        self.total_length = 0.0
        self.vocab = vocab

    def __call__(self, targets, y_hats):
        """ Calculating character error rate """
        dist, length = self._get_distance(targets, y_hats)
        self.total_dist += dist
        self.total_length += length
        return self.total_dist / self.total_length

    def _get_distance(self, targets, y_hats):
        """
        Provides total character distance between targets & y_hats
        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model
        Returns: total_dist, total_length
            - **total_dist**: total distance between targets & y_hats
            - **total_length**: total length of targets sequence
        """
        total_dist = 0
        total_length = 0

        for (target, y_hat) in zip(targets, y_hats):
            s1 = self.vocab.label_to_string(target)
            s2 = self.vocab.label_to_string(y_hat)

            dist, length = self.metric(s1, s2)

            total_dist += dist
            total_length += length

        return total_dist, total_length

    def metric(self, *args, **kwargs):
        raise NotImplementedError


class UnitErrorRate(ErrorRate):
    """
    Provides inteface of error rate calcuation.
    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, vocab) -> None:
        super(UnitErrorRate, self).__init__(vocab)

    def metric(self, s1, s2):
        """
        Computes the Unit Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        unit2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts strings)
        w1 = [chr(unit2char[w]) for w in s1.split()]
        w2 = [chr(unit2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))
