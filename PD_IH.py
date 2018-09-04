

def run(prmt, pd_inputs):
    t_ij = prmt['t_ij']
    seq0, i0 = list(map(pd_inputs.get, ['seq0', 'i0']))
    #
    iP, iM = 'p%d' % i0, 'd%d' % i0
    kP, kM = seq0[0], seq0[-1]
    detourTime1, seq1 = 1e400, None
    if len(seq0) == 2:
        seq1 = [kP, iP, iM, kM]
        detourTime1 = calc_detourTime(seq1, t_ij)
    else:
        for i in range(1, len(seq0) - 1):
            for j in range(i, len(seq0) - 1):
                seq = seq0[:]
                seq.insert(i, iP)
                seq.insert(j + 1, iM)
                detourTime = calc_detourTime(seq, t_ij)
                if detourTime < detourTime1:
                    detourTime1, seq1 = detourTime, seq
    return detourTime1, seq1


def calc_detourTime(seq, t_ij):
    detourTime = 0.0
    for i in range(len(seq) - 1):
        detourTime += t_ij[seq[i], seq[i + 1]]
    detourTime -= t_ij[seq[0], seq[-1]]
    #
    return detourTime



if __name__ == '__main__':
    pass
