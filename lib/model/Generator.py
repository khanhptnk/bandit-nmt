import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BaseGenerator(nn.Module):
    def __init__(self, generator, opt):
        super(BaseGenerator, self).__init__()
        self.generator = generator
        self.opt = opt

    def forward(self, inputs):
        return self.generator(inputs.contiguous().view(-1, inputs.size(-1)))

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        outputs = Variable(outputs.data, requires_grad=True)

        logits = outputs.contiguous().view(-1) if regression else self.forward(outputs)

        loss = criterion(logits, targets.contiguous().view(-1), weights.contiguous().view(-1))
        loss.div(normalizer).backward()
        loss = loss.data[0]

        if outputs.grad is None:
            grad_output = torch.zeros(outputs.size())
        else:
            grad_output = outputs.grad.data

        return grad_output, loss

    def predict(self, outputs, targets, weights, criterion):
        logits = self.forward(outputs)
        preds = logits.data.max(1)[1].view(outputs.size(0), -1)

        loss = criterion(logits, targets.contiguous().view(-1), weights.contiguous().view(-1)).data[0]

        return preds, loss


class MemEfficientGenerator(BaseGenerator):
    def __init__(self, generator, opt, dim=1):
        super(MemEfficientGenerator, self).__init__(generator, opt)
        self.batch_size = opt.max_generator_batches
        self.dim = dim

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        outputs_split = torch.split(outputs, self.batch_size, self.dim)
        targets_split = torch.split(targets, self.batch_size, self.dim)
        weights_split = torch.split(weights, self.batch_size, self.dim)

        grad_output = []
        loss = 0
        for out_t, targ_t, w_t in zip(outputs_split, targets_split, weights_split):
            grad_output_t, loss_t = super(MemEfficientGenerator, self).backward(
                out_t, targ_t, w_t, normalizer, criterion, regression)
            grad_output.append(grad_output_t)
            loss += loss_t

        grad_output = torch.cat(grad_output, self.dim)
        return grad_output, loss

    def predict(self, outputs, targets, weights, criterion):
        outputs_split = torch.split(outputs, self.batch_size, self.dim)
        targets_split = torch.split(targets, self.batch_size, self.dim)
        weights_split = torch.split(weights, self.batch_size, self.dim)

        preds = []
        loss = 0
        for out_t, targ_t, w_t in zip(outputs_split, targets_split, weights_split):
            preds_t, loss_t = super(MemEfficientGenerator, self).predict(
                out_t, targ_t, w_t, criterion)
            preds.append(preds_t)
            loss += loss_t

        preds = torch.cat(preds, self.dim)
        return preds, loss


