import copy 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
from layers import Conv2d, Linear

__all__ = ['masked_parameters', 'SynFlow', 'Mag', 'Taylor1ScorerAbs', 'check_sparsity', 'check_sparsity_dict', 
        'prune_model_identity', 'prune_model_custom', 'extract_mask', 'prune_conv_linear']

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def masked_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            for mask, param in zip(masks(module), module.parameters(recurse=False)):
                if param is not module.bias:
                    yield mask, param

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf 
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
    
    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params

class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)
    
    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()

class Taylor1ScorerAbs(Pruner):
    def __init__(self, masked_parameters):
        super(Taylor1ScorerAbs, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

def check_sparsity(model):

    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def check_sparsity_dict(model_dict):

    sum_list = 0
    zero_sum = 0

    for key in model_dict.keys():
        if 'mask' in key:
            sum_list = sum_list+float(model_dict[key].nelement())
            zero_sum = zero_sum+float(torch.sum(model_dict[key] == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def prune_model_identity(model):

    print('start pruning with identity mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print('identity pruning layer {}'.format(name))
            prune.Identity.apply(m, 'weight')

def prune_model_custom(model, mask_dict):

    print('start pruning with custom mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print('custom pruning layer {}'.format(name))
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def prune_conv_linear(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prune_conv_linear(model=module)

        if isinstance(module, nn.Linear):
            bias=True
            if module.bias == None:
                bias=False
            layer_new = Linear(module.in_features, module.out_features, bias)
            model._modules[name] = layer_new

        if isinstance(module, nn.Conv2d):
            layer_new = Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride)
            model._modules[name] = layer_new

    return model


