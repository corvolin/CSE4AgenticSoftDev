import numpy as np
import torch 

def compute_semantic_alignment(texts, base, model=None):
    m_distance = model.similarity(model.encode(texts), model.encode([base]))
    torch.cuda.empty_cache()
    m_distance = np.round(np.asarray(m_distance),4)
    return np.mean(m_distance.tolist()), np.max(m_distance.tolist()), np.min(m_distance.tolist()), np.std(m_distance.tolist())