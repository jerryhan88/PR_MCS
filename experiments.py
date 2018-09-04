import os.path as opath
import os
import csv, pickle
import shutil
from functools import reduce
#
from __path_organizer import exp_dpath
from _util_cython import gen_cFile
#
from problems import get_flows


def gen_prmts(machine_dpath):
    assert opath.exists(machine_dpath)
    dplym_dpath = opath.join(machine_dpath, 'dplym')
    ori_prmt_dpath = opath.join(machine_dpath, 'ori_prmt')
    for dpath in [dplym_dpath, ori_prmt_dpath]:
        assert opath.exists(dpath)
    #
    prmt_dpath = opath.join(machine_dpath, 'prmt')
    if opath.exists(prmt_dpath):
        shutil.rmtree(prmt_dpath)
    os.mkdir(prmt_dpath)
    for fn in os.listdir(dplym_dpath):
        if not fn.endswith('.pkl'):
            continue
        _, problemName = fn[:-len('.pkl')].split('_')
        sceName, _nt, _, _, _, _, _seedNum = problemName.split('-')
        #
        with open(opath.join(dplym_dpath, fn), 'rb') as fp:
            flow_oridest, task_ppdp = pickle.load(fp)
        #
        for cP in [0.90, 0.95, 0.99]:
            for alpha in [0.01, 0.001, 0.0001]:
                new_problemName = '-'.join([sceName, _nt, 'P%02d' % (cP * 100), 'a%04d' % (alpha * 10000), _seedNum])
                with open(opath.join(ori_prmt_dpath, 'prmt_%s.pkl' % problemName), 'rb') as fp:
                    ori_prmt = pickle.load(fp)
                prmt = {k: v for k, v in ori_prmt.items()}
                flows = get_flows(flow_oridest, task_ppdp)
                c_k = []
                assert type(flows) == list, type(flows)
                for _, c in flows:
                    if c == 0:
                        continue
                    c_k.append(c)
                assert len(c_k) == len(prmt['K'])
                p_k = []
                for c in c_k:
                    p_k.append(1 - (1 - alpha) ** c)
                prmt['cP'] = cP
                prmt['p_k'] = p_k
                prmt['problemName'] = new_problemName
                with open(reduce(opath.join, [prmt_dpath, 'prmt_%s.pkl' % new_problemName]), 'wb') as fp:
                    pickle.dump(prmt, fp)


def run_experiments(machine_num):
    machine_dpath = opath.join(exp_dpath, 'm%d' % machine_num)
    gen_cFile('CWL4')
    from CWL4 import run as CWL4_run
    assert opath.exists(machine_dpath)
    #
    prmt_dpath = opath.join(machine_dpath, 'prmt')
    log_dpath = opath.join(machine_dpath, 'log')
    sol_dpath = opath.join(machine_dpath, 'sol')
    for path in [log_dpath, sol_dpath]:
        if opath.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    problems_fpaths = [opath.join(prmt_dpath, fn) for fn in os.listdir(prmt_dpath)
                       if fn.endswith('.pkl')]
    problems_fpaths.sort()
    for fpath in problems_fpaths:
        with open(fpath, 'rb') as fp:
            prmt = pickle.load(fp)
        problemName = prmt['problemName']
        cwl_no = 4
        etc = {'solFilePKL': opath.join(sol_dpath, 'sol_%s_CWL%d.pkl' % (problemName, cwl_no)),
               'solFileCSV': opath.join(sol_dpath, 'sol_%s_CWL%d.csv' % (problemName, cwl_no)),
               'solFileTXT': opath.join(sol_dpath, 'sol_%s_CWL%d.txt' % (problemName, cwl_no)),
               'logFile': opath.join(log_dpath, '%s_CWL%d.log' % (problemName, cwl_no)),
               'itrFileCSV': opath.join(log_dpath, '%s_itrCWL%d.csv' % (problemName, cwl_no)),
               }
        CWL4_run(prmt, etc)




if __name__ == '__main__':
    gen_prmts(opath.join(exp_dpath, 'm0'))
    # run_experiments(0)
