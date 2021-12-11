import torch
from torch import nn
from torch.nn import functional as F, Parameter
import numpy as np


def array1d_repr(t, format='{:.3f}'):
    res = ''
    for i in range(len(t)):
        res += format.format(float(t[i]))
        if i < len(t) - 1:
            res += ', '

    return '[' + res + ']'
def array2d_repr(t, format='{:.3f}'):
    res = ''
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            res += format.format(float(t[i,j]))
            if i < t.shape[0] and j < t.shape[1] :
                res += ', '


    return '[' + res + ']'

class SteFloor(torch.autograd.Function):
    """
    Ste for floor function
    """
    @staticmethod
    def forward(ctx, a):
        return a.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ste_floor = SteFloor.apply

class SteCeil(torch.autograd.Function):
    """
    Ste for ceil function
    """
    @staticmethod
    def forward(ctx, a):
        return a.ceil()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ste_ceil = SteCeil.apply

def weight_list_to_scores(layer, layer_group_name, head_size = None):

    if layer_group_name == "W1":
        result_level1 = []
        result_level2 = []
        num_heads = layer.in_features // head_size
        for i in range(num_heads):
            group_weight = layer.weight.data[:,i * head_size : (i + 1) * head_size]
            group_weight = group_weight.reshape(-1)
            result_level2.append((group_weight ** 2).sum(0))
            result_level1.append([])
            for j in range(head_size):
                col_weight = layer.weight.data[:,i * head_size + j]
                col_weight = col_weight.reshape(-1)
                result_level1[i].append((col_weight ** 2).sum(0))
        return torch.tensor(result_level1),torch.tensor(result_level2)

    elif layer_group_name == "W3":
        result = (layer.weight.data ** 2).sum(0)
        return result.to("cpu") if result.is_cuda else result

class LeastSsum(torch.autograd.Function): ## sum of norm of least s groups
    @staticmethod
    def forward(ctx, s, weight_list):
        # vec is not sorted!
        idx = int(s.ceil().item()) + 1
        if idx <= weight_list.numel():
            vec_least_sp1 = torch.topk(weight_list, idx, largest=False, sorted=True)[0] # bottom s+1 individual values
            ctx.vec_sp1_least = vec_least_sp1[-1].item() # s+1 -th value
            return vec_least_sp1[:-1].sum() # bottom s value sum
        else:
            ctx.vec_sp1_least = weight_list.max().item()
            return weight_list.sum()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.vec_sp1_least, None

least_s_sum = LeastSsum.apply


def flops2(s,r,uvc_layers_dict,uvc_layers,head_size,ub=None,s_ub=None,r_ub=None,layer_names=None,flops_list=None):

    res = 0
    num_heads = uvc_layers['W1'][0].in_features // head_size

    for i,uvc_layer in enumerate(uvc_layers['W3']): #MLP pruning
        layer_index,col = uvc_layers_dict["s_dict"][uvc_layer]
        in_dim = uvc_layers['W3'][i].in_features - s[layer_index,col]
        out_dim = uvc_layers['W3'][i].out_features

        res += 2 * ste_floor(in_dim) * out_dim + out_dim

    for uvc_layer in uvc_layers["W1"]:
        try:
            in_dim = uvc_layer.in_features
        except Exception as e:
            print(uvc_layer)
            raise e
        out_dim = uvc_layer.out_features
        layer_index, i = uvc_layers_dict["s_dict"][uvc_layer]
        in_dim = in_dim - ste_floor(s[layer_index, i]) * head_size


        scores = weight_list_to_scores(uvc_layer, "W1", head_size)[1]
        least_s_idx = torch.topk(scores, int(s[layer_index, i].ceil().item()), largest=False, sorted=False)[1]
        for head_index in range(num_heads):
            if head_index not in least_s_idx:
                in_dim = in_dim - ste_floor(r[layer_index,head_index])
        res += 2 * in_dim * out_dim + out_dim

    return res / ub if ub is not None else res



class UVC_CP_MiniMax(nn.Module):
    def __init__(self, model, resource_fn, uvc_layers, uvc_layers_dict, head_size, num_heads, flops_list, z_init=1e-3, y_init=1e-3, p_init=1e-3,args=None):
        super(UVC_CP_MiniMax, self).__init__()
        self.model = model
        self.uvc_layers = uvc_layers
        self.uvc_layers_dict = uvc_layers_dict
        self.head_size = head_size
        n_layers = len(self.uvc_layers['W1'])
        self.n_layers = n_layers
        self.eps_decay = args.eps_decay


        self.s = Parameter(torch.zeros(n_layers,2))
        self.r = Parameter(torch.zeros(n_layers,num_heads))

        self.y = Parameter(torch.zeros(n_layers,2))
        self.p = Parameter(torch.zeros(n_layers,num_heads))
        self.y.data.fill_(y_init)
        self.p.data.fill_(p_init)
        self.z = Parameter(torch.tensor(float(z_init)))
        self.resource_fn = resource_fn


        self.enable_part_gating = args.enable_part_gating
        self.enable_block_gating = args.enable_block_gating
        self.update_gating()

        self.__least_s_norm = torch.zeros_like(self.s.data)
        self.__least_r_norm = torch.zeros_like(self.r.data)
        self.s_ub = torch.zeros_like(self.s.data).cuda()
        self.s_ub[:,0] = num_heads  # upper bound of the number of heads that can be removed
        self.s_ub[:,1] = uvc_layers["W3"][0].in_features  # upper bound of the number of neurons that can be removed

        self.r_ub = torch.zeros_like(self.r.data).cuda()
        self.r_ub[:,:] = head_size  # upper bound of dimension inside a head that can be removed
        self.num_heads = num_heads
        self.flops_list = flops_list

    def ceiled_s(self):
        return ste_ceil(self.s)
    def ceiled_r(self):
        return ste_ceil(self.r)


    def sloss1(self):
        s = self.ceiled_s()

        w_s_norm1 = torch.empty(1).cuda()

        for uvc_layer in self.uvc_layers["W1"]:
            layer_index, i = self.uvc_layers_dict["s_dict"][uvc_layer]
            temp = least_s_sum(s[layer_index,i],weight_list_to_scores(uvc_layer,"W1",self.head_size)[1]).view(-1).cuda()
            w_s_norm1 = torch.cat((w_s_norm1, temp))
        w_s_norm1 = w_s_norm1[1:w_s_norm1.shape[0]]

        result1 = self.y[:,i].data.dot(w_s_norm1)

        w_s_norm2 = torch.empty(1).cuda()
        for uvc_layer in self.uvc_layers["W3"]:
            layer_index, i = self.uvc_layers_dict["s_dict"][uvc_layer]
            temp = least_s_sum(s[layer_index,i],weight_list_to_scores(uvc_layer,"W3")).view(-1).cuda()
            w_s_norm2 = torch.cat((w_s_norm2, temp))

        w_s_norm2 = w_s_norm2[1:w_s_norm2.shape[0]]

        result2 = self.y[:,i].data.dot(w_s_norm2)

        return result1 + result2

    def rloss1(self):
        r = self.ceiled_r()

        result = 0
        for uvc_layer in self.uvc_layers["W1"]:
            w_r_norm = torch.empty(1).cuda()
            layer_index = self.uvc_layers_dict["r_dict"][uvc_layer]
            weight_list = weight_list_to_scores(uvc_layer,"W1",self.head_size)[0].cuda() # shape: [12,64]
            for head_index in range(self.num_heads):
                temp = least_s_sum(r[layer_index,head_index],weight_list[head_index,:]).view(-1)
                w_r_norm = torch.cat((w_r_norm, temp))
            w_r_norm = w_r_norm[1:w_r_norm.shape[0]]
            result += self.p[layer_index,:].data.view(-1).dot(w_r_norm)

        return result


    def run_resource_fn(self, gumbel_hard=False):
        s = self.ceiled_s()
        r = self.ceiled_r()
        rc = self.resource_fn(s, r, (self.block_skip_gating, [self.attn_skip_gating, self.mlp_skip_gating]), self.model.eps, gumbel_hard = gumbel_hard)
        return rc


    def srloss2(self, budget):
        rc = self.run_resource_fn()
        return rc - budget

    def get_least_s_norm(self):
        res = self.__least_s_norm
        s = self.ceiled_s()
        for uvc_layer in self.uvc_layers["W1"]:
            layer_index, i = self.uvc_layers_dict["s_dict"][uvc_layer]
            scores = weight_list_to_scores(uvc_layer, "W1", self.head_size)[1]

            res[layer_index,i] = torch.topk(scores, int(s[layer_index,i].ceil().item()), largest=False, sorted=False)[0].sum().item()

        for uvc_layer in self.uvc_layers["W3"]:
            layer_index, i = self.uvc_layers_dict["s_dict"][uvc_layer]
            scores = weight_list_to_scores(uvc_layer,"W3")
            res[layer_index,i] = torch.topk(scores, int(s[layer_index,i].ceil().item()), largest=False, sorted=False)[0].sum().item()
        return res

    def get_least_r_norm(self):
        res = self.__least_r_norm
        r = self.ceiled_r()
        for uvc_layer in self.uvc_layers["W1"]:
            layer_index = self.uvc_layers_dict["r_dict"][uvc_layer]
            scores = weight_list_to_scores(uvc_layer,"W1",self.head_size)[0]
            for head_index in range(self.num_heads):
                res[layer_index,head_index] = torch.topk(scores[head_index,:],int(r[layer_index,head_index].ceil().item()),largest=False,sorted=False)[0].sum().item()
        return res

    def yloss(self):
        temp = self.get_least_s_norm().cuda()
        a = 1
        return self.y[:,0].dot(temp[:,0]) + self.y[:,1].dot(temp[:,1])

    def ploss(self):
        res = 0
        temp = self.get_least_r_norm().cuda()
        for i in range(self.p.shape[0]):
            res += self.p[i,:].dot(temp[i,:])
        return res

    def zloss(self, budget):
        return self.z * (self.run_resource_fn() - budget)



    def update_gating(self):
        self.block_skip_gating  = self.model.block_skip_gating if self.enable_block_gating else None
        self.attn_skip_gating   = []    if self.enable_part_gating else None
        self.mlp_skip_gating    = []    if self.enable_part_gating else None

        if self.enable_part_gating:
            for name, p in self.model.named_parameters():
                if "attn_skip_gating" in name:
                    self.attn_skip_gating.append(p)

                if "mlp_skip_gating" in name:
                        self.mlp_skip_gating.append(p)


    def update_eps(self):
        if not self.model.enable_warmup:
            print(f"[EPS update] {self.model.eps} =====> {self.model.eps * self.eps_decay} ")
            self.model.eps = self.model.eps * self.eps_decay

    def update_sr(self):
        ## Post processing for flops calculation
        s = self.ceiled_s()
        r = self.ceiled_r()

        for uvc_layer in self.uvc_layers["W1"]:
            layer_index, col = self.uvc_layers_dict["s_dict"][uvc_layer]
            scores1, scores2 = weight_list_to_scores(uvc_layer, layer_group_name="W1", head_size=self.head_size)
            self.uvc_layer.uvc_s = int(s[layer_index, col].ceil().item()) + r[layer_index].sum()
        for uvc_layer in self.uvc_layers["W3"]:
            score = weight_list_to_scores(uvc_layer,layer_group_name="W3")
            layer_index, i = self.uvc_layers_dict["s_dict"][uvc_layer]
            uvc_layer.uvs_s = int(s[layer_index, i].ceil().item())
            self.uvc_layers["W2"][layer_index] = int(s[layer_index, i].ceil().item()) # at the same time, prune W2






def prox_w(minimax_model, optimizer):
    lr = optimizer.param_groups[0]['lr']

    s = minimax_model.ceiled_s()
    r = minimax_model.ceiled_r()

    for uvc_layer in minimax_model.uvc_layers["W1"]:
        scores1, scores2 = weight_list_to_scores(uvc_layer,layer_group_name="W1", head_size = minimax_model.head_size)

        ### First Level projection
        for head_index in range(scores1.shape[0]):
            layer_index = minimax_model.uvc_layers_dict["r_dict"][uvc_layer]
            r_cur = r[layer_index,head_index]
            least_r_idx = torch.topk(scores1[head_index,:], int(r_cur.ceil().item()), largest=False, sorted=False)[1]
            uvc_layer.weight.data[:,least_r_idx + head_index * minimax_model.head_size] /= \
                (1.0 + 2.0 * lr * minimax_model.p[layer_index,head_index].item())

        ### Second Level projection
        layer_index, i = minimax_model.uvc_layers_dict["s_dict"][uvc_layer]
        least_s_idx = torch.topk(scores2, int(s[layer_index,i].ceil().item()), largest=False, sorted=False)[1]
        for head_index in least_s_idx:
            uvc_layer.weight.data[:, head_index * minimax_model.head_size : (head_index + 1 ) * minimax_model.head_size] /= \
                (1.0 + 2.0 * lr * minimax_model.y[layer_index,i].item())

    # projection
    for uvc_layer in minimax_model.uvc_layers["W3"]:
        score = weight_list_to_scores(uvc_layer,layer_group_name="W3")
        layer_index, i = minimax_model.uvc_layers_dict["s_dict"][uvc_layer]
        least_s_idx = torch.topk(score, int(s[layer_index, i].ceil().item()), largest=False, sorted=False)[1]
        uvc_layer.weight.data[:, least_s_idx ] /= (
                    1.0 + 2.0 * lr * minimax_model.y[layer_index,i].item())


def prune_w(minimax_model,optimizer=None):
    s = minimax_model.ceiled_s()
    r = minimax_model.ceiled_r()

    for uvc_layer in minimax_model.uvc_layers["W1"]:
        layer_index, col = minimax_model.uvc_layers_dict["s_dict"][uvc_layer]
        scores1, scores2 = weight_list_to_scores(uvc_layer, layer_group_name="W1", head_size=minimax_model.head_size)

        for head_index in range(scores1.shape[0]):
            r_cur = r[layer_index,head_index]
            least_r_idx = torch.topk(scores1[head_index,:], int(r_cur.ceil().item()),largest=False, sorted=False)[1]
            uvc_layer.weight.data[:, least_r_idx + head_index * minimax_model.head_size] = 0

        least_s_idx = torch.topk(scores2, int(s[layer_index,col].ceil().item()), largest=False, sorted=False)[1]
        for head_index in least_s_idx:
            uvc_layer.weight.data[:, head_index * minimax_model.head_size : (head_index + 1 ) * minimax_model.head_size] = 0

    for uvc_layer in minimax_model.uvc_layers["W3"]:
        score = weight_list_to_scores(uvc_layer,layer_group_name="W3")
        layer_index, i = minimax_model.uvc_layers_dict["s_dict"][uvc_layer]
        least_s_idx = torch.topk(score, int(s[layer_index, i].ceil().item()), largest=False, sorted=False)[1]
        uvc_layer.weight.data[:, least_s_idx ] = 0

        minimax_model.uvc_layers["W2"][layer_index].weight.data[least_s_idx,:] = 0 # at the same time, prune W2



def prune_w_mask(minimax_model):
    s = minimax_model.ceiled_s()
    r = minimax_model.ceiled_r()

    for uvc_layer in minimax_model.uvc_layers["W1"]:
        uvc_layer.mask.data[:] = 1
        layer_index, col = minimax_model.uvc_layers_dict["s_dict"][uvc_layer]
        scores1, scores2 = weight_list_to_scores(uvc_layer, layer_group_name="W1", head_size=minimax_model.head_size)
        for head_index in range(scores1.shape[0]):
            r_cur = r[layer_index,head_index]
            least_r_idx = torch.topk(scores1[head_index,:], int(r_cur.ceil().item()),largest=False, sorted=False)[1]
            uvc_layer.mask.data[:, least_r_idx + head_index * minimax_model.head_size] = 0

        least_s_idx = torch.topk(scores2, int(s[layer_index,col].ceil().item()), largest=False, sorted=False)[1]
        for head_index in least_s_idx:
            uvc_layer.mask.data[:, head_index * minimax_model.head_size : (head_index + 1 ) * minimax_model.head_size] = 0

    for uvc_layer in minimax_model.uvc_layers["W3"]:
        uvc_layer.mask.data[:] = 1
        score = weight_list_to_scores(uvc_layer,layer_group_name="W3")
        layer_index, i = minimax_model.uvc_layers_dict["s_dict"][uvc_layer]
        least_s_idx = torch.topk(score, int(s[layer_index, i].ceil().item()), largest=False, sorted=False)[1]
        uvc_layer.mask.data[:, least_s_idx ] = 0

        minimax_model.uvc_layers["W2"][layer_index].mask.data[least_s_idx,:] = 0 # at the same time, prune W2

def proj_dual(minimax_model):
    minimax_model.y.data.clamp_(min=0.0)
    minimax_model.p.data.clamp_(min=0.0)
    minimax_model.z.data.clamp_(min=0.0)


def calc_flops(s, r, uvc_layers_dict, uvc_layers, head_size, s_ub, r_ub, flops_list, gating, eps, full_model_flops=None, use_gumbel=False, gumbel_hard=True):
    (embed_macs, total_macs)  = flops_list


    if full_model_flops is not None:
        total_macs = torch.Tensor(total_macs)
        # total_macs = [layers * 6]  6 = (macs_proj*s_0, macs_qk*s_0, macs_v*r, macs_w1*r*r, macs_mlp*s_1, macs_mlp*s_1)
        s_ratio = (s_ub - s) / s_ub
        s_ratio = s_ratio.clamp(0.0, 1.0)
        attn_proj = r_ub.clone().sum(1)   # embed_dim

        for uvc_layer in  uvc_layers["W1"]:
            layer_index, col =  uvc_layers_dict["s_dict"][uvc_layer]
            scores1, scores2 = weight_list_to_scores(uvc_layer, layer_group_name="W1", head_size= head_size)
            least_s_idx = torch.topk(scores2, int( s[layer_index,col].ceil().item()), largest=False, sorted=False)[1]

            # First prune multi-heads using s[0]
            attn_proj[layer_index] -=  s[layer_index, 0] *  head_size

            # Then prune per-head weights using r
            for i, head_index in enumerate(range(scores1.shape[0]), 0):
                # if the head needs to be pruned, just delete the whole head
                if i in least_s_idx:
                    continue
                else:
                    attn_proj[layer_index] -=  r[layer_index, head_index]

        r_ratio = attn_proj /  r_ub.sum(1)
        r_ratio = r_ratio.clamp(0.0, 1.0)


        # Multiply with block gating function

        distrib1 = 1
        if block_gating is not None:
            if use_gumbel:
                distrib1 = torch.nn.functional.gumbel_softmax(block_gating, tau=0.5, hard=gumbel_hard, eps=1e-10, dim=1)[:,1].cuda()
            else:
                tmp = block_gating**2
                distrib1 = (tmp/(tmp + eps))[:, 1].cuda()

        total_macs = total_macs.cuda()
        total_macs = (total_macs.transpose(0, 1)*distrib1).transpose(0,1)



        macs = embed_macs   \
                                + (total_macs[:, 0] * s_ratio[:, 0]).sum() \
                                + (total_macs[:, 1] * s_ratio[:, 0]).sum() \
                                + (total_macs[:, 2] * r_ratio).sum() \
                                + (total_macs[:, 3] * r_ratio * r_ratio).sum() \
                                + (total_macs[:, 4] * s_ratio[:, 1]).sum() \
                                + (total_macs[:, 5] * s_ratio[:, 1]).sum()

        flops = macs*2 / full_model_flops




    else:

        macs = embed_macs + torch.Tensor(total_macs).sum()
        flops = macs*2
    return flops



class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print('\n\n=> Using a preset learning rate schedule:')
        print(decay_schedule)
        print("\n")
        self.for_once = True

    def __call__(self, optimizer, iteration, lr_name="zlr"):
        for param_group in optimizer.param_groups:
            if lr_name in param_group:
                lr = self.decay_schedule.get(iteration, param_group[lr_name])
                if param_group[lr_name] != lr:
                    print(f"====> The learning rate paramater \"{lr_name}\" changed from {param_group[lr_name]} to {lr}")
                    param_group[lr_name] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr