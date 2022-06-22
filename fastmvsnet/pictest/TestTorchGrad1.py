import sys
sys.path.append('/home/pic/downloads/FastMVSNet/fastmvsnet')

import torch

if __name__ == '__main__':
    # a = torch.rand(10, 10, requires_grid=True)
    a = torch.ones(10, requires_grad=True)
    print(a)
    b = 2 * a

    c = b * 3

    loss = c.sum()

    print(loss)

    loss.backward()
    # a = 
    print(1)