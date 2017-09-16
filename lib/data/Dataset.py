from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import lib


class Dataset(object):
    def __init__(self, data, batchSize, cuda, eval=False):
        self.src = data["src"]
        self.tgt = data["tgt"]
        self.pos = data["pos"]
        assert(len(self.src) == len(self.tgt))
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.eval = eval

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(lib.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(self.src[index*self.batchSize:(index+1)*self.batchSize],
            include_lengths=True)

        tgtBatch = self._batchify(self.tgt[index*self.batchSize:(index+1)*self.batchSize])

        # within batch sort by decreasing length.
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b):
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.eval)
            return b

        return (wrap(srcBatch), lengths), wrap(tgtBatch), indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt, self.pos))
        random.shuffle(data)
        self.src, self.tgt, self.pos = zip(*data)

    def restore_pos(self, sents):
        sorted_sents = [None] * len(self.pos)
        for sent, idx in zip(sents, self.pos):
          sorted_sents[idx] = sent
        return sorted_sents
