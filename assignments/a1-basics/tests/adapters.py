"""Adapter hooks for connecting student implementations to the A1 tests.

Implement these functions; the assignment tests intentionally expect
NotImplementedError until the student wires in their own code.
"""


def _unimplemented(*args, **kwargs):
    raise NotImplementedError


def run_linear(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_embedding(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_swiglu(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_scaled_dot_product_attention(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_multihead_self_attention(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_multihead_self_attention_with_rope(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_rope(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_transformer_block(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_transformer_lm(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_rmsnorm(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_silu(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_get_batch(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_softmax(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_cross_entropy(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_gradient_clipping(*args, **kwargs): return _unimplemented(*args, **kwargs)
def get_adamw_cls(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_get_lr_cosine_schedule(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_save_checkpoint(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_load_checkpoint(*args, **kwargs): return _unimplemented(*args, **kwargs)
def get_tokenizer(*args, **kwargs): return _unimplemented(*args, **kwargs)
def run_train_bpe(*args, **kwargs): return _unimplemented(*args, **kwargs)
