"""
CareER-GPT | Recalcula las métricas del modelo de lenguaje a partir de las rutas generadas en Colab.
La coherencia exige que los tres programas del ranking aparezcan en la ruta y en el mismo orden, como la
define la metodología. ROUGE-L, BERTScore, intervalos y pruebas siguen el cuaderno de Colab.
"""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import json, re, numpy as np, pandas as pd
from scipy import stats
from bert_score import score as bscore

SEED = 42
G = pd.read_csv('resultados/generaciones.csv').fillna('')
LL = json.load(open('resultados/resultados_llm.json'))
ref = G.referencia.tolist(); instr = G.instruccion.tolist()


def toks(s): return re.findall(r'\w+', s.lower())
def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]; dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1]); prev = cur
    return dp[-1]
def rouge_l(c, r, beta=1.2):
    a, b = toks(c), toks(r)
    if not a or not b: return 0.0
    l = lcs(a, b); p, rc = l / len(a), l / len(b)
    return 0.0 if l == 0 else (1 + beta**2) * p * rc / (rc + beta**2 * p)
def programas(i):
    m = re.search(r'Probabilidades estimadas de ingreso:(.*?)\. Genera', i, re.S)
    return [x.group(1).strip() for x in re.finditer(r'([^;]+?) (\d+(?:\.\d+)?) %', m.group(1))] if m else []
def coherente(gen, i):
    """1 si los tres programas del ranking aparecen en la ruta y en el mismo orden."""
    pr = programas(i); g = gen.upper(); pos = [g.find(p.upper()) for p in pr]
    return int(len(pr) == 3 and all(q >= 0 for q in pos) and pos == sorted(pos))
def evaluar(gen):
    rl = np.array([rouge_l(a, b) for a, b in zip(gen, ref)])
    _, _, F = bscore(gen, ref, lang='es', verbose=False); bf = F.numpy()
    co = np.array([coherente(a, b) for a, b in zip(gen, instr)])
    return dict(rougeL=rl, bertscore=bf, coherencia=co, combinada=(rl + bf + co) / 3)
def ic(x, B=1000):
    rng = np.random.default_rng(SEED); b = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

CB, CS, CH = 'LLM base sin ajuste', 'LLM + LoRA sin componente tabular', 'CareER-GPT híbrido'
M = {CB: evaluar(G.base.tolist()), CS: evaluar(G.solo_llm.tolist()), CH: evaluar(G.hibrido.tolist())}
LL['metricas_colab'] = LL.get('metricas_colab', LL['metricas'])  # se conservan las originales del cuaderno
LL['pruebas_colab'] = LL.get('pruebas_colab', LL['pruebas'])
LL['metricas'] = {k: {m: ic(v[m]) for m in ('rougeL', 'bertscore', 'coherencia', 'combinada')} for k, v in M.items()}
H = M[CH]; res = {}
k_ = int(H['coherencia'].sum()); bt = stats.binomtest(k_, len(H['coherencia']), 0.80, alternative='greater')
res['H3'] = dict(coherencia=k_ / len(H['coherencia']), n=len(H['coherencia']), p=bt.pvalue)
for rival in (CB, CS):
    dif = H['combinada'] - M[rival]['combinada']; sw = stats.shapiro(dif)
    pr = stats.ttest_rel(H['combinada'], M[rival]['combinada'], alternative='greater') if sw.pvalue > 0.05 \
        else stats.wilcoxon(H['combinada'], M[rival]['combinada'], alternative='greater')
    mejora = (H['combinada'].mean() - M[rival]['combinada'].mean()) / M[rival]['combinada'].mean() * 100
    res['H4_vs_' + rival] = dict(mejora_pct=float(mejora), shapiro_p=float(sw.pvalue),
                                 prueba='t pareada' if sw.pvalue > 0.05 else 'Wilcoxon',
                                 estadistico=float(pr.statistic), p=float(pr.pvalue))
LL['pruebas'] = res
LL['coherencia_definicion'] = 'tres programas del ranking presentes y en el mismo orden'
json.dump(LL, open('resultados/resultados_llm.json', 'w'), indent=2, ensure_ascii=False)
print(json.dumps({k: {m: round(v[m][0], 3) for m in v} for k, v in LL['metricas'].items()}, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
