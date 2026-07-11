import os
import numpy as np
import faiss
import torch

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    
    # Convert input to numpy array
    x_np = x.cpu().numpy()
    
    # Search with FAISS
    D_np, I_np = index.search(x_np, k)
    
    # Copy results back to PyTorch tensors
    D.copy_(torch.from_numpy(D_np))
    I.copy_(torch.from_numpy(I_np))
    
    torch.cuda.synchronize()
    return D, I

def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(512 * 1024 * 1024)  # Set temp memory to 512MB
    return res


def _env_truthy(name):
    return os.environ.get(name, '').lower() in ('1', 'true', 'yes')


def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    """k-NN search; default CPU ``IndexFlatL2`` to avoid cublas(13) abort when Faiss GPU mismatches CUDA/cuBLAS.

    If Faiss-GPU matches your environment, set ``FAISS_USE_GPU=1`` for ``GpuIndexFlatL2`` (requires valid ``res``).
    """
    assert xb.device == xq.device

    nq, d = xq.size()
    nb, d2 = xb.size()
    assert d2 == d

    if D is None:
        D = torch.empty((nq, k), dtype=torch.float32, device=xb.device)
    if I is None:
        I = torch.empty((nq, k), dtype=torch.int64, device=xb.device)

    xb_np = np.ascontiguousarray(xb.detach().float().cpu().numpy(), dtype=np.float32)
    xq_np = np.ascontiguousarray(xq.detach().float().cpu().numpy(), dtype=np.float32)

    use_gpu = _env_truthy('FAISS_USE_GPU') and metric == faiss.METRIC_L2
    if use_gpu:
        if res is None:
            res = faiss.StandardGpuResources()
            res.setTempMemory(512 * 1024 * 1024)
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = torch.cuda.current_device()
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        index.add(xb_np)
        D_np, I_np = index.search(xq_np, k)
    else:
        index = faiss.IndexFlatL2(d)
        index.add(xb_np)
        D_np, I_np = index.search(xq_np, k)

    D.copy_(torch.from_numpy(D_np))
    I.copy_(torch.from_numpy(I_np))
    return D, I

def index_init_gpu(ngpus, feat_dim):
    flat_config = []
    for i in range(ngpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    indexes = [faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i]) for i in range(ngpus)]
    index = faiss.IndexShards(feat_dim)
    for sub_index in indexes:
        index.add_shard(sub_index)
    index.reset()
    return index

def index_init_cpu(feat_dim):
    return faiss.IndexFlatL2(feat_dim)
