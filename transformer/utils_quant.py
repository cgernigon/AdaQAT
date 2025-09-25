# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import math


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None

class SymLsqQuantizer(torch.autograd.Function):
    """
        Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = torch.tensor(-2 ** (num_bits - 1), device=input.device)
        Qp = torch.tensor(2 ** (num_bits - 1) - 1, device=input.device)

        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp + 0.0000000001)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp

        q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class LsqStepSize(nn.Parameter):
    def __init__(self, tensor):
        #super(LsqStepSize, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        # print('Stepsize initialized to %.6f' % self.data.item())
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        # input: everthing needed to initialize step_size
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)

class QuantizeLinear(nn.Linear):

    def __init__(self,  *kargs,bias=True, config = None):
        super(QuantizeLinear, self).__init__(*kargs,bias=True)
        self.weight_clip_val = LsqStepSize(torch.tensor(1.0, device='cuda'))
        self.input_clip_val = LsqStepSize(torch.tensor(1.0, device='cuda'))

        self.float_W_width = torch.FloatTensor([4])
        self.float_A_width = torch.FloatTensor([4])
        self.W_bitwidth = torch.tensor([4])
        self.A_bitwidth = torch.tensor([4])
        self.weight_only = False
        self.quant_weight = self.weight.detach().clone()

        self.Wits = {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0} # Weight counter
        self.Wconvergence = False # flag set when weights bit-width layer converges
        self.Aits = {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0} # Activation counter
        self.Aconvergence = False # flag set when activation bit-width layer converges
        self.CEA = 0.0
        self.bitops = 0
        self.computeBitOps = False

    def forward(self, input):
        if self.computeBitOps:
            self.bitOps()

        weight = SymLsqQuantizer.apply(self.weight, self.weight_clip_val, self.W_bitwidth, True)
        input_q = SymLsqQuantizer.apply(input, self.input_clip_val, self.A_bitwidth, True)

        self.quant_weight = weight.detach().clone()

        out = nn.functional.linear(input_q, weight)

        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        self.CEA = torch.nn.functional.cross_entropy(input_q.data, input.data).abs().item()
        self.OAq = input_q.data

        return out

    def bitOps(self):
        self.bitops = self.W_bitwidth.item()*self.A_bitwidth.item()*self.in_features*self.out_features

class QuantizeEmbedding(nn.Embedding):

    def __init__(self,  *kargs,padding_idx=None, config = None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = 2
        self.layerwise = False
        self.weight_clip_val = LsqStepSize(torch.tensor(1.0, device='cuda'))

    def forward(self, input):
        weight = SymLsqQuantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)

        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out
