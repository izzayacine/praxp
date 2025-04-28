#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## lmpxp.py
##
##  Created on: Feb 23, 2023
##

import subprocess
import platform
import sys
import os
#import random
import resource
import getopt
import signal
from decimal import Decimal
from tqdm import tqdm

import numpy as np


from pyapproxmc import Counter

sys.path.append('./RFxpl')
from xrf import XRF, Dataset
from xrf import RFBreiman, RFSklearn
from xrf import SATEncoder, SATExplainer
from pysat.solvers import Solver

from six.moves import range
import six
import math




# datasets={
#         'kr_vs_kp':5, 
#         'mofn_3_7_10': 6, 
#         'parity5+5': 8
#     }

# # hard datasets
# datasets.update({
#           'heart-c':5,
#           'twonorm':3,
#           ##'spectf':5,        
#           'ionosphere':5,
#           'segmentation':4
#          })

# models = [f'./models/{data}/{data}_nbestim_100_maxdepth_{depth}.mod.pkl' 
#           for data, depth in datasets.items()]

# cat_data = False
# abdxp = [f'../../infxp/scripts/results/abd/{data}.log' for data in datasets]
# #datafiles = [f'../../xprf/RFxpl-experiments/bench/datasets/{data}/{data}.csv' for data in datasets]
# datafiles = [f'./datasets/{data}/{data}.csv' for data in datasets]

#
#==============================================================================
# categorical datasets
datasets = {
    'adult': 8,
    'vote':5,
    'tic_tac_toe': 7
}

# hard datasets
datasets.update({
    'german': 8,
    'soybean': 5,    
    'agaricus_lepiota': 5
})

cat_data = True

model_dir = './Classifiers/RFs'
models = [f'{model_dir}/{data}/{data}_nbestim_100_maxdepth_{depth}.mod.pkl' 
         for data, depth in datasets.items()]

abdxp = [f'./results/RFs/abd/{data}.log' for data in datasets]
datafiles = [f'datasets/categorical/{data}/{data}.csv' for data in datasets]

#
#==============================================================================
class TimeExceededException(Exception):
    pass
def signal_handler(signum, frame):
    raise TimeExceededException("Timed out!")


#
# ===================================================================================

def sample_fn(doms, inst, n_examples):
    examples = np.zeros((n_examples,len(inst)), dtype=np.float32)
    for i in range(len(inst)):
        if inst[i] is None:
            examples[:, i] = np.random.choice(doms[i], size=n_examples)
        else:
            examples[:, i] = np.full((n_examples,), inst[i])
    
    return examples        
#
# ===================================================================================
class ProbAXplainer(object):
    """
        Probablistic abductive explainer
    """
    def __init__(self, cls, xpl, featdoms, ohe,  etype='sat', 
                 mc=False, alg='del', verbose=1, timelimit=60):
        self.cls = cls
        self.xpl = xpl
        self.etype = etype
        self.doms = featdoms
        self.data_encoder = ohe
        #
        self.alg = alg
        self.approxmc = mc # if yes then ApproxMC otherwise Monte-Carlo Sampling
        self.verb = verbose
        self._timeout = timelimit

        
    def drset(self, inst, expl, threshold=0.95, beta=0.1):
        """
            xpl: explainer
            inst: data input vector
            threshold: minimum precision
            1 - beta: confidence
            return a LmPAXp
        """
        self.eps = (1. - threshold)/2.
        assert (self.eps > 0.)
        self.delta = beta/len(expl)
        
        orginst = np.array(inst)
        inst = self.data_encoder(np.array(inst))[0]
        
        if expl is None:
            expl = self.xpl.explain(inst, xtype='abd', etype=self.etype)
        
        self.xpl.encode(inst, self.etype) 
    
    
        inpvals = self.xpl.readable_data(inst)
        preamble = []
        for f, v in zip(self.xpl.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)
                    
        inps = self.xpl.ffnames # input (feature value) variables
    
        assert (self.xpl.f.attr_names == self.xpl.ffnames)
    
    
        x = SATExplainer(self.xpl.enc, inps, preamble, self.xpl.class_names, self.verb)
        x.prepare_selectors(np.asarray(inst))
        x.assums = sorted(set(x.assums))
        hypos = [h for h in x.assums if (x.sel2fid[h] in expl) ]
    
        av = {}
    
        # continuous data
        # use Bitvector encoding for continuous data
        for f, intvs in six.iteritems(x.enc.ivars):
            if not len(intvs):
                continue
            if len(intvs) <= 4:
                av[f] = intvs
            else:
                n = math.ceil(math.log2(len(intvs)))
                av[f] = [x.enc.newVar(f'{f}_a{j}') for j in range(n)]
                # encode z <=> bitvector[k]
                for k, z in enumerate(intvs):
                    bv = [int(j) for j in np.binary_repr(k, width=n)]
                    # reverse the boolean encoding, so the left most var is the least bit
                    bv.reverse()
                    bv = [av[f][j]*(2*b-1) for j,b in enumerate(bv)]
                    x.enc.cnf.append([z]+[-a for a in bv])
                    x.enc.cnf.extend([[-z, a] for a in bv])
                # deactivate unused bitvectors in the encoding, i.e. ~bv
                for j in range(len(intvs), 2**n):
                    bv = [int(j) for i in format(j,'b').zfill(n)]
                    bv.reverse()
                    bv = [av[f][i]*(2*b-1) for i,b in enumerate(bv)]
                    x.enc.cnf.append([-a for a in bv])
            
            
    
        # categorical data
        for f, featvars in six.iteritems(x.enc.categories):
            av[f] = featvars # use OHE for categorical features 
    #         n = math.ceil(math.log2(len(featvars)))
    #         av[f] = [x.enc.newVar(f'{f}_a{j}') for j in range(n)]
        
    #         for k, z in enumerate(featvars):
    #             bv = [int(j) for j in np.binary_repr(k, width=n)]
    #             # reverse the boolean encoding, so the left most var is the least bit
    #             bv.reverse()
    #             bv = [av[f][j]*(2*b-1) for j,b in enumerate(bv)]
    #             x.enc.cnf.append([z]+[-a for a in bv])
    #             x.enc.cnf.extend([[-z, a] for a in bv])
    #         for j in range(len(featvars), 2**n):
    #             bv = [int(j) for i in format(j,'b').zfill(n)]
    #             bv.reverse()
    #             bv = [av[f][i]*(2*b-1) for i,b in enumerate(bv)]
    #             x.enc.cnf.append([-a for a in bv])    
        
        def deletionMC(core):
            i = 0
            pr = 1.0
            # hypos.reverse() ???
            while i<len(core):
                prec = self.precision(x, core[:i]+core[i+1:], av, orginst, True)
                if prec >= threshold:
                    core = core[:i]+core[i+1:]
                    pr = prec
                else:
                    i = i+1
            return core, pr
        
        def delete(core):
            ##
            mc_pr = 1.0
            prec = 1.
            while prec > threshold:
                to_test = [core[:i]+core[i+1:] for i in range(len(core))]
                probs = [self.precision(x, test, av, orginst) for test in to_test]
                i = np.argmax(probs)
                #prec = self.precision(x, core[:i]+core[i+1:], av, approxmc=True)
                prec = max(probs)
                if prec >= threshold:
                    core = core[:i]+core[i+1:]
                    mc_pr = prec
            
            return core, mc_pr        
        
        
#         def progress(to_test):
#             prec = 0.
#             core = []
#             while (prec < threshold):
#                 if len(to_test) == 1: 
#                     core.append(to_test[-1])
#                     prec = 1.
#                     break
#                 probs = [self.precision(x, core+[h], av, orginst) for h in to_test]
#                 core.append(to_test[np.argmax(probs)])
#                 del to_test[np.argmax(probs)]
#                 #prec = self.precision(x, core, av, approxmc=True)
#                 prec = max(probs)
#             return core, prec         
        
# #         def progress2(to_test):
# #             prec = 0.
# #             core = []
# #             while (prec < threshold):
# #                 if len(to_test) == 1: 
# #                     core.append(to_test[-1])
# #                     prec = 1.
# #                     break
# #                 prec = self.precision(x, core+[to_test[-1]], av, orginst)
# #                 core.append(to_test[-1])
# #                 del to_test[-1]
# #                 #prec = self.precision(x, core, av, approxmc=True)
# #             return core, prec
        
        

        core, prc = delete(hypos)
            
        expl = [x.sel2fid[h] for h in core if h>0]  
    
        return expl, prc


    
    def precision(self, x, hypos, projv, inst=None, approxmc=False):
        """
            estimate accuracy of the epxlanation: hypos
        """
        univ = [x.sel2fid[h] for h in x.assums if (h not in hypos)]
    
        dom = []
        for j in univ:
            f = 'f{0}'.format(j)
            if f in x.enc.intvs:
                n = len(x.enc.intvs[f])
            else:
                assert f in x.enc.categories
                n = len(x.enc.categories[f])
            if n:
                dom.append(n)
            
        #fspace = np.prod(dom, dtype=np.int64)
    
        if approxmc:
            samp_vars = [a for j in univ for a in projv[f'f{j}']]
        
            mc = Counter(seed=2157, epsilon=0.8, delta=0.2, verbosity=0)
            # pass a CNF formula
            for cl in x.enc.cnf:
                mc.add_clause(cl)
            for h in hypos:
                mc.add_clause([h])
        
            try:
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(timelimit)  # timeout in seconds
                
                sols, hashes =  mc.count(samp_vars)
                res = (sols*2**hashes)
    
            except TimeExceededException:
                print('  mcount timeout:', self._timeout)
                res = 0
                
            # calculate the proba epsilon = Pr(k(x)≠c)
            prob = res
            for d in dom:
                prob /= d
                
            if self.verb>1:
                print("#projected vars:",len(samp_vars))
                print('mc=', res, "  prob={0:.3f}".format( prob))                
            
        else:
            # Monte Carlo Sampling
            n_samples = math.ceil((1/(2*self.eps**2))*math.log(1/self.delta))
            # print(self.eps, math.log(1/self.delta), 1/(2*self.eps**2))
            assert (inst is not None)
            data_point = [v if not(j in univ) else None for j,v in enumerate(inst)]
            samples = sample_fn(self.doms, data_point, n_samples)
            label = self.cls.predict(self.data_encoder(inst))
            prob = sum(self.cls.predict(self.data_encoder(samples))==label)/n_samples
            if self.verb>1:
                print('nof sample=', n_samples)
                print(f'prob={prob:3f}')
            prob = 1. - prob
        
        
        if self.verb>1:
            print(f'prec={1. - prob:.3f}')
    
        return 1. - prob

#
#==============================================================================
def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'a:ht:x:X:v',
                                   ['alg=',
                                    'help',
                                    'tau=',
                                    'sample=',
                                    'axp=',
                                    'timelimit=',
                                    'verb'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    # init 
    alg = 'del'
    verb = 0
    tau = 0.05 # (%)
    axp = None
    sample = None
    timelimit= 60

    for opt, arg in opts:
        if opt in ('-t', '--tau'):
            tau = float(str(arg))
            assert (tau > 0.0)
        elif opt in ('-a', '--alg'):
            alg = str(arg)
            assert alg in ['del', 'prog']
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-x', '--sample'):
            sample = str(arg)
        elif opt in ('-X', '--axp'):
            axp = eval(str(arg)) 
        elif opt in ('--timelimit'):
            timelimit = int(str(arg)) 
            assert timelimit > 0
        elif opt in ('-v', '--verb'):
            verb += 1
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)
    
    tau = (100. - 100.*tau) / 100
    
    return tau, alg, sample, axp, timelimit, verb, args    
#
#==============================================================================
def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options]  RF-model')
    print('Options:')
    print('        -h, --help')
    print('        -a, --alg=<string>      Algo to compute LmPAXp del/prog-based (default = del)')
    print('        -t, --tau=<float>       Tolerance threshold (default = 0.05)')
    print('        -x, --sample=<csv>      Explain the prediction of the given data input')
    print('        -X, --axp=<csv>         Initial set (W)AXp to compute LmPAXp (default = None)')
    print('        -v, --verb              Be verbose (show comments)')
    
#
#==============================================================================
if __name__ == '__main__':
    
    xtype = 'abd'
    etype = 'sat'
    
    nof_samples = 100
    
    tau, alg, sample, axp, timelimit, verb, files = parse_options() 
#     if len(files) == 0:
#         print('.pkl file is missing!')
#         exit()     
    

    pathfile = './results/RFs/abd/'
    for data, model, filename, dfile in zip(datasets, models, abdxp, datafiles):
        print(data)
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            lines = [line.strip() for line in lines]
            samples = filter(lambda x: x.startswith('inst:'), lines)
            samples = [s.split(':')[1]  for s in samples] 
            axps = filter(lambda x: x.startswith('expl:'), lines)
            axps = [eval(x.split(':')[1]) for x in axps]
    
        # load model
        cls = RFSklearn(from_file=model)
        xrf = XRF(cls, cls.feature_names, cls.targets)
        
        dataset = Dataset(filename=dfile, use_categorical=cat_data)
        if cat_data:
            xrf.ffnames = dataset.m_features_
            xrf.readable_data = lambda x: dataset.readable_sample(dataset.transform_inverse(x)[0])        
        
        
        featdoms = [np.unique(dataset.X[:,i]) for i in range(dataset.X.shape[1])]
        probxpl = ProbAXplainer(cls, xrf, featdoms, dataset.transform, etype='sat',  
                                mc=False, alg=alg, verbose=verb, timelimit=timelimit)
        
        
        results = []
        times = []
        lengths = []
        prec = []
        
        for i, (inst, axp) in enumerate(zip(samples, axps)):
            results.append({})
            results[i]['inst'] = inst
            inst = [float(v.strip()) for v in inst.split(',')]
            
            #if cat_data:
            #    inst = dataset.transform(np.array(inst))[0]
            
            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime  
            
            
            expl, pr =  probxpl.drset(inst, axp)
            #expl, pr =  probxpl.drset(orginst, inst, axp, tau=0.95, beta=0.1)
    
            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
            times.append(time)
            lengths.append(len(expl))
            prec.append(pr)
        
            results[i]['nvars'] = xrf.enc.nVars
            results[i]['nclauses'] = xrf.enc.nClauses
            results[i]['expl'] = expl
            results[i]['len'] = len(expl)
            results[i]['len%'] = (len(expl)/len(axp))*100.
            results[i]['pr'] = f'{pr:.2f}'
            results[i]['time'] = f'{time:.2f}'
            del xrf.enc
            #del xrf.x 
            
            print('i=',i, '  axp:',len(axp), ' expl:',len(expl), f'  {time:.2f}s')
            
            if nof_samples == i+1:
                break
            
               
        #break
        # save the results
        
#         output =  f'results/RFs/LmPAXp/{alg}/{data}.log'
#         try:
#             os.stat(f'results/RFs/LmPAXp/{alg}')
#         except:
#             os.makedirs(f'results/RFs/LmPAXp/{alg}') 
        
#         with open(output, 'w') as fp:
#             fp.write(f'{data}\n')
#             fp.write('************\n')
#             for i in range(len(results)):
#                 fp.write(f"inst: {results[i]['inst']}\n")
#                 fp.write(f"nof vars: {results[i]['nvars']}\n")
#                 fp.write(f"nof clauses: {results[i]['nclauses']}\n")
#                 fp.write(f"expl: {results[i]['expl']}\n")
#                 fp.write(f"expl len: {results[i]['len']}\n")
#                 fp.write(f"expl len(%): {results[i]['len%']:.1f}\n")
#                 fp.write(f"prec: {results[i]['pr']}\n")
#                 fp.write(f"time: {results[i]['time']}\n")
#                 fp.write('\n')
#             m, mx, avg = min(lengths), max(lengths), sum(lengths)/len(lengths)
#             fp.write(f"min expl: {m}\n")
#             fp.write(f"max expl: {mx}\n")
#             fp.write(f"avg expl: {avg:.1f}\n")
#             fp.write('\n')
#             m, mx, avg = min(times), max(times), sum(times)/len(times)
#             fp.write(f"min time: {m:.2f}\n")
#             fp.write(f"max time: {mx:.2f}\n")
#             fp.write(f"avg time: {avg:.2f}\n")            
#             fp.write('\n')
#             fp.write(f'threshold: {tau}\n')
#             m, mx, avg = min(prec), max(prec), sum(prec)/len(prec)
#             fp.write(f"min prec: {m:.2f}\n")
#             fp.write(f"max prec: {mx:.2f}\n")
#             fp.write(f"avg prec: {avg:.2f}\n") 
            
        #break
    
    print()    
    print('done.')    