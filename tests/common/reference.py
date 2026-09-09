"""Independent Float64 oracle for the equations in the task statement.

Only zero-RoPE fixtures are evaluated: the statement does not provide the
complete nonzero-RoPE fusion formula.
"""
import numpy as np
from tests.common.tensor_io import decode_logical

REFERENCE_VERSION = "spec-kl-v2-double-scale-zero-rope"


def logsumexp(value, axis=None, keepdims=False):
    maximum = np.max(value, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(value - maximum).sum(axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def indexer_kl_backward(query_index, key_index, weights, sparse_indices, target_selected):
    qi = np.asarray(query_index, dtype=np.float64)
    ki = np.asarray(key_index, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    idx = np.asarray(sparse_indices, dtype=np.int64)
    target = np.asarray(target_selected, dtype=np.float64)
    target = target / target.sum()
    s = qi @ ki[idx].T
    relu = np.maximum(s, 0.0)
    logits = w @ relu
    log_prediction = logits - logsumexp(logits)
    prediction = np.exp(log_prediction)
    active = target > 0
    loss = np.sum(target[active] * (np.log(target[active]) - log_prediction[active]))
    di = prediction - target
    ds = w[:, None] * di[None, :] * (s > 0)
    dk = np.zeros_like(ki)
    np.add.at(dk, idx, ds.T @ qi)
    return {"dQueryIndex": ds @ ki[idx], "dKeyIndex": dk,
            "dWeights": relu @ di, "loss": np.asarray([loss])}


def canonical_arrays(arrays, case):
    qshape = (case.b, case.s1, case.n1, case.d_query)
    shapes = {
        "query": qshape, "key": (case.b, case.s2, 1, case.d_query),
        "queryIndex": (case.b, case.s1, case.nidx1, case.d_query_index),
        "keyIndex": (case.b, case.s2, 1, case.d_query_index),
        "weights": (case.b, case.s1, case.nidx1),
        "queryRope": (case.b, case.s1, case.n1, case.d_rope),
        "keyRope": (case.b, case.s2, 1, case.d_rope),
    }
    result = {name: decode_logical(arrays[name], case.input_dtypes()[name]).reshape(shape)
              for name, shape in shapes.items()}
    result["sparseIndices"] = arrays["sparseIndices"].reshape(case.b, case.s1, case.k)
    for name in ("softmaxMax", "softmaxSum"):
        if case.statistics_layout == "BGS" and case.layout != "TND":
            result[name] = arrays[name].transpose(0, 2, 1)
        else:
            result[name] = arrays[name].reshape(case.b, case.s1, case.n1)
    return result


def reference_backward(arrays, case):
    data = canonical_arrays(arrays, case)
    if np.any(data["queryRope"]) or np.any(data["keyRope"]):
        raise ValueError("Nonzero RoPE has no confirmed reference formula in this suite")
    qi, ki, w = (data[name] for name in ("queryIndex", "keyIndex", "weights"))
    dq = np.zeros(qi.shape, dtype=np.float64)
    dk = np.zeros(ki.shape, dtype=np.float64)
    dw = np.zeros(w.shape, dtype=np.float64)
    loss = 0.0
    scale = float(case.scale_value)
    for b in range(case.b):
        qlength = case.actual_query_lengths()[b]
        klength = case.actual_key_lengths()[b]
        for t in range(qlength):
            indices = data["sparseIndices"][b, t]
            valid = (indices >= 0) & (indices < klength) & (indices <= t + klength - qlength)
            indices = indices[valid]
            if indices.size == 0:
                raise ValueError("The statement does not define an empty selected distribution")
            q = data["query"][b, t].astype(np.float64)
            key = data["key"][b, indices, 0].astype(np.float64)
            scores = (q @ key.T) * scale
            adjusted = scores - data["softmaxMax"][b, t, :, None]
            adjusted -= np.log(data["softmaxSum"][b, t, :, None].astype(np.float64))
            log_mass = logsumexp(adjusted, axis=0)
            target = np.exp(log_mass - logsumexp(log_mass))
            selected_ki = ki[b, indices, 0].astype(np.float64)
            row_qi = qi[b, t].astype(np.float64)
            row_w = w[b, t].astype(np.float64)
            similarity = row_qi @ selected_ki.T
            relu = np.maximum(similarity, 0.0)
            logits = row_w @ relu
            log_predicted = logits - logsumexp(logits)
            predicted = np.exp(log_predicted)
            log_target = log_mass - logsumexp(log_mass)
            loss += np.sum(target * (log_target - log_predicted))
            di = predicted - target
            ds = row_w[:, None] * di[None, :] * (similarity > 0.0)
            dq[b, t] = ds @ selected_ki
            dw[b, t] = relu @ di
            np.add.at(dk[b, :, 0], indices, ds.T @ row_qi)
    return {name: value.reshape(case.output_shapes()[name]) for name, value in
            (("dQueryIndex", dq), ("dKeyIndex", dk), ("dWeights", dw),
             ("loss", np.asarray([loss], dtype=np.float64)))}
