import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

import utils.data_utils
import utils.globals as uglobals


def run_pc_kci(mmlu_path, specs_path, out_name='debug', alpha=0.2):
    np_out, model_names, col_names = utils.data_utils.data_to_np(mmlu_path, specs_path, no_nan=True)
    
    cg = pc(np_out, alpha=alpha, indep_test='kci', polyd=3, est_width='median')
    pyd = GraphUtils.to_pydot(cg.G, labels=col_names)
    pyd.write_png(f'{uglobals.FIGS_DIR}/{out_name}_alpha{alpha}.png')
    pyd.write_raw(f'{uglobals.DOT_DIR}/{out_name}_alpha{alpha}.dot')
    return

def run_ges(mmlu_path, specs_path, out_name='debug', score='local_score_marginal_general'):
    np_out, model_names, col_names = utils.data_utils.data_to_np(mmlu_path, specs_path, no_nan=True, no_discrete=True)
    
    cg = ges(np_out, score_func=score, maxP=4)
    pyd = GraphUtils.to_pydot(cg['G'], labels=col_names)
    pyd.write_png(f'{uglobals.FIGS_DIR}/{out_name}_{score}.png')
    pyd.write_raw(f'{uglobals.DOT_DIR}/{out_name}_{score}.dot')
    return

if __name__ == '__main__':
    mmlu_path = uglobals.MMLU_PATH
    specs_path = uglobals.SPECS_PATH
    run_pc_kci(mmlu_path, specs_path, out_name='pc_kci')
    # run_ges(mmlu_path, specs_path, out_name='ges')

    # Grouped
    # mmlu_path = uglobals.MMLU_GROUPED_PATH
    # run_pc_kci(mmlu_path, specs_path, out_name='pc_kci_grouped')
    # run_ges(mmlu_path, specs_path, out_name='ges_grouped')