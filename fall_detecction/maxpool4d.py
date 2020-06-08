import torch
from torch.autograd import Variable


def maxpool4d(corr4d_hres, k_size=4):
    slices = []
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    sl = corr4d_hres[:, 0, i::k_size, j::k_size, k::k_size, l::k_size].unsqueeze(0)
                    slices.append(sl)

    slices = torch.cat(tuple(slices), dim=1)
    corr4d, max_idx = torch.max(slices, dim=1, keepdim=True)
    max_l = torch.fmod(max_idx, k_size)
    max_k = torch.fmod(max_idx.sub(max_l).div(k_size), k_size)
    max_j = torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size), k_size)
    max_i = max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d, max_i, max_j, max_k, max_l)


input_tensor = Variable(torch.rand(1, 1, 2, 2, 3, 4))
print(input_tensor)
print(maxpool4d(input_tensor, 2))
