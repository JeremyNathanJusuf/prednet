from evaluate import _calculate_mse
import numpy as np
import torch 

def test_mse():
    pred = np.ones((10, 9, 3, 128, 128))
    pred[5:,...] = 0.0
    gt = np.zeros((10, 9, 3, 128, 128))
    
    mse = _calculate_mse(pred, gt)
    print("mse", mse)
    assert abs(mse - 0.5) < 1e-6
    print("test mse successful")
    
if __name__ == '__main__':
    test_mse()