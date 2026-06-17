import numpy as np
import torch 

def compute_semantic_alignment(texts, base, model=None):
    m_distance = model.similarity(model.encode(texts), model.encode([base]))
    torch.cuda.empty_cache()
    m_distance = np.round(np.asarray(m_distance),6)
    return m_distance.tolist()