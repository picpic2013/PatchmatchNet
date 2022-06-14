import sys
from time import time
sys.path.append('/home/pic/downloads/FastMVSNet/fastmvsnet')

import torch
import numpy as np

from picutils import PICTimer

if __name__ == '__main__':
    timer = PICTimer.getTimer("aaa").startTimer()
    # print(timer.startTimer())
    timer.showTime("bb")

    # num = 5
    num = 58982400
    
    for _ in range(10):
        a = torch.rand(num, 1, 2).to('cuda:1')
        b = torch.rand(num, 2, 1).to('cuda:1')

        timer.showTime("start")

        torch.cuda.synchronize()
        bb = b.permute(0, 2, 1)
        # cc = (a * bb).sum(dim=2)
        cc = torch.sum((a * bb), dim=2)
        # print(np.allclose((a@b).cpu().numpy(), cc.cpu().numpy()))

        torch.cuda.synchronize()
        timer.showTime("end_" + str(_))

    print((a * bb).shape)
    timer.summary()