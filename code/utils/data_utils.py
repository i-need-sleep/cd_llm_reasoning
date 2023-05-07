import numpy as np
import pandas as pd
import json

def data_to_np(mmlu_path, specs_path, no_nan=False, no_discrete=False):
    # Piece together MMLU and specification data, return a np array, col names and row names
    out = pd.read_excel(specs_path)
    for col in out.columns:
        if no_nan and out[col].isnull().any():
            del out[col]
    out = out.to_dict(orient='list')
    
    if no_discrete:
        out = {
            'Model name': out['Model name'],
            'Parameters (M)': out['Parameters (M)']
        }
    
    # Build placeholders for performances
    included = []
    with open(mmlu_path, 'r') as f:
        mmlu = json.load(f)
    for idx, (model_name, perfs) in enumerate(mmlu.items()):
        if idx == 0:
            for subject in perfs.keys():
                out[subject] = [0 for _ in range(len(mmlu))]

        # Resolve model idx
        model_idx = out['Model name'].index(model_name.replace('_', '/').replace('cambridgeltl/magic/mscoco', 'cambridgeltl/magic_mscoco').replace('bigscience/bloom-7b1', 'decapoda-research/llama-7b-hf'))
        
        for subject, perf in perfs.items():
            out[subject][model_idx] = perf
    
    model_names = out['Model name']
    del out['Model name']

    out = pd.DataFrame(out)
    np_out = out.to_numpy()
    col_names = [c for c in out.columns]
    return np_out, model_names, col_names

if __name__ == '__main__':
    mmlu_path = '../../results/aggregated/aggregated.json'
    specs_path = '../../data/llm_specs.xlsx'
    out = data_to_np(mmlu_path, specs_path, no_nan=True, no_discrete=True)
    print(out)
    np.isnat