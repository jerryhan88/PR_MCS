import os.path as opath
import multiprocessing
import datetime, time
import csv, pickle
import numpy as np
from gurobipy import *
#
from _util_cython import gen_cFile

prefix = 'PD_IH'
gen_cFile(prefix)
from PD_IH import run as PD_IH_run


NUM_CORES = multiprocessing.cpu_count()
EPSILON = 0.00001


def write_log(fpath, contents):
    with open(fpath, 'a') as f:
        logContents = '\n\n'
        logContents += '======================================================================================\n'
        logContents += '%s\n' % str(datetime.datetime.now())
        logContents += '%s\n' % contents
        logContents += '======================================================================================\n'
        f.write(logContents)


def itr2file(fpath, contents=[]):
    if not contents:
        if opath.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = ['itrNum',
                      'eliCpuTime', 'eliWallTime',
                      'numCols', 'numTB',
                      'relObjV', 'selBC', 'selBC_RC', 'new_RC_BC']
            writer.writerow(header)
    else:
        with open(fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(contents)


def res2file(fpath, objV, gap, eliCpuTime, eliWallTime):
    with open(fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['objV', 'Gap', 'eliCpuTime', 'eliWallTime']
        writer.writerow(header)
        writer.writerow([objV, gap, eliCpuTime, eliWallTime])


def generate_RMP(prmt, add_inputs):
    C, p_c, e_ci = list(map(add_inputs.get, ['C', 'p_c', 'e_ci']))
    includeBNB = True if 'inclusiveC' in add_inputs else False
    #
    bB, T = list(map(prmt.get, ['bB', 'T']))
    #
    # Define decision variables
    #
    RMP = Model('RMP')
    q_c = {}
    for c in range(len(C)):
        q_c[c] = RMP.addVar(vtype=GRB.BINARY, name="q[%d]" % c)
    RMP.update()
    #
    # Define objective
    #
    obj = LinExpr()
    for c in range(len(C)):
        obj += p_c[c] * q_c[c]
    RMP.setObjective(obj, GRB.MAXIMIZE)
    #
    # Define constrains
    #
    taskAC = {}
    for i in T:  # eq:taskA
        taskAC[i] = RMP.addConstr(quicksum(e_ci[c][i] * q_c[c] for c in range(len(C))) <= 1,
                                  name="taskAC[%d]" % i)
    numBC = RMP.addConstr(quicksum(q_c[c] for c in range(len(C))) <= bB,
                              name="numBC")
    if includeBNB:
        inclusiveC, exclusiveC = list(map(add_inputs.get, ['inclusiveC', 'exclusiveC']))
        C_i0i1 = {}
        for i0, i1 in set(inclusiveC).union(set(exclusiveC)):
            for c in range(len(C)):
                if i0 in C[c] and i1 in C[c]:
                    if (i0, i1) not in C_i0i1:
                        C_i0i1[i0, i1] = []
                    C_i0i1[i0, i1].append(c)
        #
        for i, (i0, i1) in enumerate(inclusiveC):
            RMP.addConstr(quicksum(q_c[b] for b in C_i0i1[i0, i1]) >= 1,
                              name="mIC[%d]" % i)
        for i, (i0, i1) in enumerate(exclusiveC):
            RMP.addConstr(quicksum(q_c[b] for b in C_i0i1[i0, i1]) <= 0,
                              name="mEC[%d]" % i)
    RMP.update()
    #
    return RMP, q_c, taskAC, numBC

def estimate_RP(prmt, cwl_inputs, c, i0):
    K, p_k, _delta = list(map(prmt.get, ['K', 'p_k', '_delta']))
    s_ck = cwl_inputs['s_ck']
    unrealizableProb, seqs = 1.0, []
    for k in K:
        seq0 = s_ck[c, k]
        detourTime, seq1 = PD_IH_run(prmt, {'seq0': seq0, 'i0': i0})
        if detourTime <= _delta:
            unrealizableProb *= (1 - p_k[k])
        seqs.append(seq1)
    #
    realizableProb = 1 - unrealizableProb
    return realizableProb, seqs


def LS_run(prmt, cwl_inputs):
    T, cB_P, cP = [prmt.get(k) for k in ['T', 'cB_P', 'cP']]
    #
    pi_i, mu = [cwl_inputs.get(k) for k in ['pi_i', 'mu']]
    C, sC, c0 = [cwl_inputs.get(k) for k in ['C', 'sC', 'c0']]
    #
    Ts0 = C[c0]
    rc_Ts1_seqs = []
    for i0 in T:
        if i0 in Ts0:
            continue
        Ts1 = Ts0[:] + [i0]
        if frozenset(tuple(Ts1)) in sC:
            continue
        if len(Ts1) > cB_P:
            continue
        rp, seqs = estimate_RP(prmt, cwl_inputs, c0, i0)
        if rp >= cP:
            vec = [0 for _ in range(len(T))]
            for i in Ts1:
                vec[i] = 1
            rc = len(Ts1) - (np.array(vec) * np.array(pi_i)).sum() - mu
            #
            rc_Ts1_seqs.append([rc, Ts1, seqs])
    #
    return rc_Ts1_seqs


def run(prmt, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    if 'TimeLimit' not in etc:
        etc['TimeLimit'] = 1e400
    etc['startTS'] = startCpuTime
    etc['startCpuTime'] = startCpuTime
    etc['startWallTime'] = startWallTime
    itr2file(etc['itrFileCSV'])
    #
    cwl_inputs = {}
    T, cB_M, cB_P, K = [prmt.get(k) for k in ['T', 'cB_M', 'cB_P', 'K']]
    #
    C, sC, p_c, e_ci = [], set(), [], []
    TB = set()
    s_ck = {}
    for i in T:
        c = len(C)
        iP, iM = 'p%d' % i, 'd%d' % i
        for k in K:
            kP, kM = 'ori%d' % k, 'dest%d' % k
            s_ck[c, k] = [kP, iP, iM, kM]
        Ts = [i]
        C.append(Ts)
        sC.add(frozenset(tuple(Ts)))
        #
        p_c.append(0)
        #
        vec = [0 for _ in range(len(T))]
        vec[i] = 1
        e_ci.append(vec)
    #
    cwl_inputs['C'] = C
    cwl_inputs['sC'] = sC
    cwl_inputs['p_c'] = p_c
    cwl_inputs['e_ci'] = e_ci
    cwl_inputs['TB'] = TB
    cwl_inputs['s_ck'] = s_ck
    #
    RMP, q_c, taskAC, numBC = generate_RMP(prmt, cwl_inputs)
    #
    counter, is_terminated = 0, False
    while True:
        if len(C) == len(T) ** 2 - 1:
            break
        LRMP = RMP.relax()
        LRMP.setParam('Threads', NUM_CORES)
        LRMP.setParam('OutputFlag', False)
        LRMP.optimize()
        if LRMP.status == GRB.Status.INFEASIBLE:
            logContents = 'Relaxed model is infeasible!!\n'
            logContents += 'No solution!\n'
            write_log(etc['logFile'], logContents)
            #
            LRMP.write('%s.lp' % prmt['problemName'])
            LRMP.computeIIS()
            LRMP.write('%s.ilp' % prmt['problemName'])
            assert False
        #
        pi_i = [LRMP.getConstrByName("taskAC[%d]" % i).Pi for i in T]
        mu = LRMP.getConstrByName("numBC").Pi
        cwl_inputs['pi_i'] = pi_i
        cwl_inputs['mu'] = mu
        #
        c0, minRC = -1, 1e400
        for rc, c in [(LRMP.getVarByName("q[%d]" % c).RC, c) for c in range(len(C))]:
            Ts = C[c]
            if c in TB:
                continue
            if len(Ts) == cB_P:
                continue
            if rc < -EPSILON:
                TB.add(c)
                continue
            if rc < minRC:
                minRC = rc
                c0 = c
        if c0 == -1:
            break
        cwl_inputs['c0'] = c0
        #
        rc_Ts1_seqs = LS_run(prmt, cwl_inputs)
        rc_Ts1 = [[o[0], o[1]] for o in rc_Ts1_seqs]
        if time.clock() - etc['startTS'] > etc['TimeLimit']:
            break
        #
        eliCpuTimeP, eliWallTimeP = time.clock() - etc['startCpuTime'], time.time() - etc['startWallTime']
        itr2file(etc['itrFileCSV'], [counter, '%.2f' % eliCpuTimeP, '%.2f' % eliWallTimeP,
                                     len(cwl_inputs['C']), len(cwl_inputs['TB']),
                                     '%.2f' % LRMP.objVal, C[c0], '%.2f' % minRC, str(rc_Ts1)])
        if len(rc_Ts1_seqs) == 0:
            TB.add(c0)
        else:
            is_updated = False
            for rc, Ts1, seqs in rc_Ts1_seqs:
                if rc < 0:
                    continue
                is_updated = True
                vec = [0 for _ in range(len(T))]
                for i in Ts1:
                    vec[i] = 1
                if sum(vec) < cB_M:
                    p = 0
                else:
                    p = rc + (np.array(vec) * np.array(pi_i)).sum() + mu
                C, p_c, e_ci, sC = list(map(cwl_inputs.get, ['C', 'p_c', 'e_ci', 'sC']))
                e_ci.append(vec)
                p_c.append(p)
                #
                col = Column()
                for i in range(len(T)):
                    if e_ci[len(C)][i] > 0:
                        col.addTerms(e_ci[len(C)][i], taskAC[i])
                col.addTerms(1, numBC)
                #
                q_c[len(C)] = RMP.addVar(obj=p_c[len(C)], vtype=GRB.BINARY, name="q[%d]" % len(C), column=col)
                for k in K:
                    s_ck[len(C), k] = seqs[k]
                C.append(Ts1)
                sC.add(frozenset(tuple(Ts1)))
                RMP.update()
                #
            if not is_updated:
                TB.add(c0)
        if len(C) == len(TB):
            break
        counter += 1
    #
    # Handle termination
    #
    RMP.setParam('Threads', NUM_CORES)
    RMP.optimize()
    #
    if etc and RMP.status != GRB.Status.INFEASIBLE:
        assert 'solFilePKL' in etc
        assert 'solFileCSV' in etc
        assert 'solFileTXT' in etc
        #
        q_c = [RMP.getVarByName("q[%d]" % c).x for c in range(len(C))]
        chosenC = [(C[c], '%.2f' % q_c[c]) for c in range(len(C)) if q_c[c] > 0.5]
        with open(etc['solFileTXT'], 'w') as f:
            endCpuTime, endWallTime = time.clock(), time.time()
            eliCpuTime = endCpuTime - etc['startCpuTime']
            eliWallTime = endWallTime - etc['startWallTime']
            logContents = 'Summary\n'
            logContents += '\t Cpu Time: %f\n' % eliCpuTime
            logContents += '\t Wall Time: %f\n' % eliWallTime
            logContents += '\t ObjV: %.3f\n' % RMP.objVal
            logContents += '\t Gap: %.3f\n' % RMP.MIPGap
            logContents += 'Chosen bundles\n'
            logContents += '%s\n' % str(chosenC)
            f.write(logContents)
            f.write('\n')
        #
        res2file(etc['solFileCSV'], RMP.objVal, RMP.MIPGap, eliCpuTime, eliWallTime)
        #
        _q_c = {c: RMP.getVarByName("q[%d]" % c).x for c in range(len(C))}
        sol = {
            'C': C, 'p_c': p_c, 'e_ci': e_ci,
            #
            'q_c': _q_c}
        with open(etc['solFilePKL'], 'wb') as fp:
            pickle.dump(sol, fp)


if __name__ == '__main__':
    pass