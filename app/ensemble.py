from typing import Dict

def average_probs(prob_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Promedio simple de probabilidades a través de modelos.
    prob_dict: {modelo: {clase: prob}}
    """
    if not prob_dict:
        return {}
    classes = list(next(iter(prob_dict.values())).keys())
    out = {c: 0.0 for c in classes}
    for _, pd in prob_dict.items():
        for c in classes:
            out[c] += float(pd[c])
    n = len(prob_dict)
    for c in classes:
        out[c] /= n
    return out