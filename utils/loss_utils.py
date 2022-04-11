# -*- coding:utf-8 -*-
from torch.autograd import Function


def share_loss(span_outputs, type_outputs, loss_funct, layers=3):
    # ((batch_size, seq, dim), ...) # Layer-0, 1, ...
    loss = 0.0
    for i in range(layers):
        loss += loss_funct(span_outputs[i], type_outputs[i])

    return loss

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        
        return output, None