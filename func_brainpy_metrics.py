import jax
import brainpy.math as bm

# define some metrics based on jax_metric
def cosine_similarity(target, preds, axis=-1, epsilon=1e-05):
    target = target / bm.maximum(bm.linalg.norm(target, axis=axis, keepdims=True), bm.sqrt(epsilon))
    preds  = preds  / bm.maximum(bm.linalg.norm(preds,  axis=axis, keepdims=True), bm.sqrt(epsilon))
    return  bm.sum(target * preds, axis=axis)

def correlation_similarity(target, preds, axis=-1, epsilon=1e-05):
    target = bm.normalize(target, axis=axis, epsilon=epsilon)
    preds  = bm.normalize(preds,  axis=axis, epsilon=epsilon)
    return bm.mean(target * preds, axis=axis)

correlation_similarity_flatt=lambda x,y: correlation_similarity(bm.flatten(x),bm.flatten(y))
correlation_similarity_batch=jax.vmap(correlation_similarity_flatt)

cosine_similarity_flatt=lambda x,y: cosine_similarity(bm.flatten(x),bm.flatten(y))
cosine_similarity_batch=jax.vmap(cosine_similarity_flatt)
