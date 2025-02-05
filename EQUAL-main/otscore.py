import torch
import json
from tqdm import tqdm
import numpy as np
import io
import base64
import os

def ipot_WD(a1, a2, C, beta=2, max_iter=1000, L=1, use_path=True, return_map=True, return_loss=True, epsilon=1e-8):
    ns = len(a1)
    nt = len(a2)
    
    v = np.ones(nt)
    u = np.ones(ns)

    P = np.ones((ns, nt)) / (ns * nt)

    K = np.exp(-C / beta)
    if return_loss:
        loss = []
    for outer_i in range(max_iter):
        Q = K * P  # Q = K ⊙ P

        if not use_path:
            v = np.ones(nt)
            u = np.ones(ns)

        for i in range(L):
            Qv = np.matmul(Q, v) + epsilon
            u = a1 / Qv
            Qtu = np.matmul(Q.T, u) + epsilon
            v = a2 / Qtu 

        P = np.outer(u, v) * Q
        if return_loss:
            W = np.sum(P * C) 
            loss.append(W)

    if return_loss:
        if return_map:
            return P, loss
        else:
            return loss
    else:
        if return_map:
            return P
        else:
            return None

def compute_cost_matrix_2d(a_points, b_points, metric='euclidean'):
    if metric == 'euclidean':
        C = np.linalg.norm(a_points[:, np.newaxis, :] - b_points[np.newaxis, :, :], axis=2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return C

def get_optimal_score(sampled_data):
    file_path_validation = './data/100_gsm8k.pt'

    if os.path.exists(file_path_validation):
        validation_set = torch.load(file_path_validation, weights_only=True).cpu()
        validation_set = validation_set.to(dtype=torch.float32) 
        validation_set_np = validation_set.numpy()

            
    result = []
    for embed in sampled_data:
        buffer = io.BytesIO(base64.b64decode(embed['embs']))
        buffer = np.load(buffer, allow_pickle=False)
        result.append(buffer)
        result_tensor = torch.tensor(np.array(result)).cpu()
        result_tensor = result_tensor.to(dtype=torch.float32) 
        result_tensor_np = result_tensor.numpy() 
    C = compute_cost_matrix_2d(validation_set_np, result_tensor_np, metric='euclidean')
    n = validation_set_np.shape[0]
    m = result_tensor_np.shape[0]
    a_weights = np.ones(n) / n
    b_weights = np.ones(m) / m 

    gamma, loss = ipot_WD(a_weights, b_weights, C, beta=1, max_iter=500, L=10, use_path=True, return_map=True, return_loss=True)

    return 1-loss[-1]-0.12

