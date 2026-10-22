import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import json
celdas = []
def md(t): celdas.append({'cell_type':'markdown','metadata':{},'source':t})
def code(t): celdas.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':t})

md("""# CareER-GPT: ajuste con LoRA y validación de la arquitectura híbrida
Objetivos específicos 3 y 4. Ejecutar en Google Colab con **Entorno de ejecución > Cambiar tipo > GPU T4**.

Al terminar, descarga `resultados_llm.zip` y envíalo para completar el borrador.

Tiempo aproximado en T4: entre 2 y 3 horas con la configuración por defecto.""")
code("""!pip -q install -U transformers peft bitsandbytes accelerate datasets bert-score""")
code("""# ===== CONFIGURACIÓN =====
MODELO = 'Qwen/Qwen2.5-1.5B-Instruct'   # LLM de código abierto con buen desempeño en español
MAX_TRAIN = 6000     # pares de entrenamiento por condición (subir si hay tiempo)
MAX_VAL = 300
N_EVAL = 300         # pares de prueba para generación y evaluación
MAXLEN = 512
R, ALPHA, DROPOUT = 16, 32, 0.05
EPOCAS, LR = 1, 2e-4
SEED = 42
SYS = ('Eres un orientador vocacional de la Universidad Nacional del Altiplano de Puno. '
       'Responde en español, de forma clara, a partir del perfil y de las probabilidades estimadas.')""")
code("""# Sube CareER_Dataset.jsonl cuando lo pida
import json, re, random, numpy as np, torch
from google.colab import files
subido = files.upload()
datos = [json.loads(l) for l in open('CareER_Dataset.jsonl', encoding='utf8')]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tr = [d for d in datos if d['split'] == 'entrenamiento']; va = [d for d in datos if d['split'] == 'validacion']; te = [d for d in datos if d['split'] == 'prueba']
random.shuffle(tr); random.shuffle(va); random.shuffle(te)
tr, va, te = tr[:MAX_TRAIN], va[:MAX_VAL], te[:N_EVAL]
print(len(datos), 'pares |', len(tr), 'entrenamiento |', len(va), 'validación |', len(te), 'prueba')

def sin_probs(instr):
    # versión del perfil SIN la salida del componente tabular (condición solo LLM)
    return re.sub(r' Probabilidades estimadas de ingreso:.*?\\. Genera', ' Genera', instr)
def programas(instr):
    m = re.search(r'Probabilidades estimadas de ingreso: (.*?)\\. Genera', instr)
    return [re.sub(r' \\d+\\.\\d %$', '', x.strip()) for x in m.group(1).split(';')] if m else []
print(sin_probs(te[0]['instruction'])[:300]); print(programas(te[0]['instruction']))""")
code("""from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments,
                          Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
tok = AutoTokenizer.from_pretrained(MODELO)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
def prompt(instr):
    return tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': instr}],
                                   tokenize=False, add_generation_prompt=True)
def tokenizar(ej, con_probs):
    p = prompt(ej['instruction'] if con_probs else sin_probs(ej['instruction']))
    ids_p = tok(p, add_special_tokens=False)['input_ids']
    ids = tok(p + ej['output'] + tok.eos_token, add_special_tokens=False, truncation=True, max_length=MAXLEN)['input_ids']
    lab = ([-100] * len(ids_p) + ids[len(ids_p):])[:len(ids)]   # la pérdida solo se calcula sobre la respuesta
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': lab}
def cargar_base():
    m = AutoModelForCausalLM.from_pretrained(MODELO, quantization_config=BNB, device_map='auto', torch_dtype=torch.float16)
    m.config.use_cache = False
    return m""")
code("""import time, gc
def entrenar(con_probs, nombre):
    tok.padding_side = 'right'
    base = cargar_base()
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias='none', task_type='CAUSAL_LM',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    model = get_peft_model(base, cfg)
    entren = sum(p.numel() for p in model.parameters() if p.requires_grad); total = sum(p.numel() for p in model.parameters())
    dtr = Dataset.from_list([tokenizar(e, con_probs) for e in tr]); dva = Dataset.from_list([tokenizar(e, con_probs) for e in va])
    args = TrainingArguments(output_dir=nombre, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                             per_device_eval_batch_size=8, num_train_epochs=EPOCAS, learning_rate=LR,
                             lr_scheduler_type='cosine', warmup_ratio=0.03, logging_steps=10, eval_strategy='steps',
                             eval_steps=50, save_strategy='no', fp16=True, optim='paged_adamw_8bit',
                             gradient_checkpointing=True, report_to='none', seed=SEED)
    t0 = time.time()
    tr_ = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    tr_.train()
    info = dict(parametros_entrenables=int(entren), parametros_totales=int(total), porcentaje=100 * entren / total,
                minutos=(time.time() - t0) / 60, historial=tr_.state.log_history)
    model.save_pretrained(nombre + '_adaptador')
    return model, info""")
code("""# Condición híbrida: LLM + LoRA entrenado CON las probabilidades del componente tabular
modelo_h, info_h = entrenar(True, 'hibrido')
print({k: v for k, v in info_h.items() if k != 'historial'})""")
code("""@torch.no_grad()
def generar(model, instrucciones, lote=12, max_new=200):
    tok.padding_side = 'left'; model.eval(); model.config.use_cache = True; salidas = []
    for i in range(0, len(instrucciones), lote):
        enc = tok([prompt(x) for x in instrucciones[i:i + lote]], return_tensors='pt', padding=True,
                  add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        salidas += tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f'{min(i + lote, len(instrucciones))}/{len(instrucciones)}', end=' ')
    return [s.strip() for s in salidas]
instr_con = [e['instruction'] for e in te]; instr_sin = [sin_probs(e['instruction']) for e in te]
gen_hibrido = generar(modelo_h, instr_con)
with modelo_h.disable_adapter():          # mismo modelo sin adaptadores = LLM base sin ajuste
    gen_base = generar(modelo_h, instr_con)
print(); print(gen_hibrido[0]); print('---'); print(gen_base[0])""")
code("""# Condición solo LLM: LoRA entrenado SIN la salida del componente tabular
del modelo_h; gc.collect(); torch.cuda.empty_cache()
modelo_s, info_s = entrenar(False, 'solo_llm')
gen_solo = generar(modelo_s, instr_sin)
print(); print(gen_solo[0])""")
code("""from bert_score import score as bscore
from scipy import stats
ref = [e['output'] for e in te]
def toks(s): return re.findall(r'\\w+', s.lower())
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
def coherente(gen, instr):
    pr = programas(instr); g = gen.upper()
    if not pr or pr[0].upper() not in g: return 0
    pos = [g.find(p.upper()) for p in pr]; return int(all(pos[0] <= q for q in pos if q >= 0))
def evaluar(gen):
    rl = np.array([rouge_l(g, r) for g, r in zip(gen, ref)])
    P, Rc, F = bscore(gen, ref, lang='es', verbose=False); bf = F.numpy()
    co = np.array([coherente(g, i) for g, i in zip(gen, instr_con)])
    comb = (rl + bf + co) / 3
    return dict(rougeL=rl, bertscore=bf, coherencia=co, combinada=comb)
M = {'LLM base sin ajuste': evaluar(gen_base), 'LLM + LoRA sin componente tabular': evaluar(gen_solo),
     'CareER-GPT híbrido': evaluar(gen_hibrido)}
def ic(x, B=1000):
    rng = np.random.default_rng(SEED); b = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
tabla = {k: {m: ic(v[m]) for m in v} for k, v in M.items()}
for k, v in tabla.items(): print(k, {m: round(x[0], 4) for m, x in v.items()})""")
code("""H = M['CareER-GPT híbrido']; res = {}
# H3: coherencia contextual del modelo ajustado mayor a 0.80 (prueba binomial exacta unilateral)
k_ = int(H['coherencia'].sum()); bt = stats.binomtest(k_, len(H['coherencia']), 0.80, alternative='greater')
res['H3'] = dict(coherencia=k_ / len(H['coherencia']), n=len(H['coherencia']), p=bt.pvalue)
# H4: el híbrido supera en al menos 10 % a cada modelo individual en la métrica combinada
for rival in ['LLM base sin ajuste', 'LLM + LoRA sin componente tabular']:
    dif = H['combinada'] - M[rival]['combinada']
    sw = stats.shapiro(dif[:5000])
    prueba = stats.ttest_rel(H['combinada'], M[rival]['combinada'], alternative='greater') if sw.pvalue > 0.05 \\
        else stats.wilcoxon(H['combinada'], M[rival]['combinada'], alternative='greater')
    mejora = (H['combinada'].mean() - M[rival]['combinada'].mean()) / M[rival]['combinada'].mean() * 100
    res['H4_vs_' + rival] = dict(mejora_pct=float(mejora), shapiro_p=float(sw.pvalue),
                                 prueba='t pareada' if sw.pvalue > 0.05 else 'Wilcoxon',
                                 estadistico=float(prueba.statistic), p=float(prueba.pvalue))
print(json.dumps(res, indent=1, ensure_ascii=False))""")
code("""import matplotlib.pyplot as plt, pandas as pd, zipfile
def curva(info, nombre):
    h = info['historial']; a = [(x['step'], x['loss']) for x in h if 'loss' in x]; b = [(x['step'], x['eval_loss']) for x in h if 'eval_loss' in x]
    return a, b
fig, ax = plt.subplots(figsize=(7, 3.6))
for info, et, c in [(info_h, 'Híbrido', '#2E75B6'), (info_s, 'Solo LLM', '#C55A11')]:
    a, b = curva(info, et)
    ax.plot(*zip(*a), color=c, alpha=.5, lw=1, label=f'{et}: entrenamiento'); ax.plot(*zip(*b), color=c, lw=2, marker='o', ms=3, label=f'{et}: validación')
ax.set_xlabel('Paso de entrenamiento'); ax.set_ylabel('Pérdida de entropía cruzada'); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig('fig_perdida.png', dpi=220)
pd.DataFrame({'instruccion': instr_con, 'referencia': ref, 'base': gen_base, 'solo_llm': gen_solo, 'hibrido': gen_hibrido}).to_csv('generaciones.csv', index=False)
out = dict(modelo=MODELO, config=dict(r=R, alpha=ALPHA, dropout=DROPOUT, epocas=EPOCAS, lr=LR, max_train=MAX_TRAIN, n_eval=N_EVAL, maxlen=MAXLEN),
           entrenamiento={'hibrido': {k: v for k, v in info_h.items() if k != 'historial'}, 'solo_llm': {k: v for k, v in info_s.items() if k != 'historial'}},
           historial={'hibrido': info_h['historial'], 'solo_llm': info_s['historial']}, metricas=tabla, pruebas=res,
           gpu=torch.cuda.get_device_name(0))
json.dump(out, open('resultados_llm.json', 'w'), indent=2, ensure_ascii=False)
with zipfile.ZipFile('resultados_llm.zip', 'w') as z:
    for f in ['resultados_llm.json', 'generaciones.csv', 'fig_perdida.png']: z.write(f)
files.download('resultados_llm.zip')""")
nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
      'colab': {'provenance': [], 'gpuType': 'T4'}}, 'cells': celdas}
for c in nb['cells']:
    c['source'] = [l + '\n' for l in c['source'].split('\n')]; c['source'][-1] = c['source'][-1].rstrip('\n')
json.dump(nb, open('colab/CareER-GPT_LoRA_Colab.ipynb', 'w'), indent=1, ensure_ascii=False)
print('cuaderno OK', len(celdas), 'celdas')
