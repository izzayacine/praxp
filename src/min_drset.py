#!/usr/bin/env python3
#-*- coding:utf-8 -*-

#
#==============================================================================
from __future__ import print_function
#from _fileio import FileObject
import collections
import os
import sys


import getopt
import resource
import math 
from functools import reduce

import numpy as np


from z3 import Real, Int, RealVal, IntVal, Bool, Bools, BoolVal 
from z3 import And, Or, Not, Sum, Product, Implies, If, PbGe
from z3 import Optimize, Solver, sat, unsat

#
#==============================================================================
class Node():
    def __init__(self, feat='', vals=[]):
        self.feat = feat
        self.vals = vals

#
#==============================================================================
class DecisionTree():
    def __init__(self, from_file=None, from_fp=None,
            mapfile=None, medges=False, verbose=0):

        self.verbose = verbose
        self.medges = medges and mapfile != None

        self.nof_nodes = 0
        self.nof_terms = 0
        self.root_node = None
        self.terms = []
        self.nodes = {}
        self.paths = {}
        self.feats = []
        self.feids = {}
        self.fdoms = {}
        self.fvmap = {}

        # OHE mapping
        OHEMap = collections.namedtuple('OHEMap', ['dir', 'opp'])
        self.ohmap = OHEMap(dir={}, opp={})

        if from_file:
            self.from_file(from_file)
        elif from_fp:
            self.from_fp(from_fp)

        if mapfile:
            self.parse_mapping(mapfile)
        else:  # no mapping is given
            for f in self.feats:
                for v in self.fdoms[f]:
                    self.fvmap[tuple([f, v])] = '{0}={1}'.format(f, v)

        if self.medges:
            self.convert_to_multiedges()

    def from_file(self, filename):
        """
            Parse a DT file.
        """

        with open(filename, mode='r') as fp:    
            self.from_fp(fp)

    def from_fp(self, fp):
        """
            Get the tree from a file pointer.
        """

        lines = fp.readlines()

        # filtering out comment lines (those that start with '#')
        lines = list(filter(lambda l: not l.startswith('#'), lines))

        # number of nodes
        self.nof_nodes = int(lines[0].strip())

        # root node
        self.root_node = int(lines[1].strip())

        # number of terminal nodes (classes)
        self.nof_terms = len(lines[3][2:].strip().split())

        # the ordered list of terminal nodes
        self.terms = {}
        for i in range(self.nof_terms):
            nd, _, t = lines[i + 4].strip().split()
            self.terms[int(nd)] = t #int(t)

        # finally, reading the nodes
        self.nodes = collections.defaultdict(lambda: Node(feat='', vals={}))
        self.feats = set([])
        self.fdoms = collections.defaultdict(lambda: set([]))
        for line in lines[(4 + self.nof_terms):]:
            # reading the tuple
            nid, fid, fval, child = line.strip().split()

            # inserting it in the nodes list
            self.nodes[int(nid)].feat = fid
            self.nodes[int(nid)].vals[int(fval)] = int(child)

            # updating the list of features
            self.feats.add(fid)

            # updaing feature domains
            self.fdoms[fid].add(int(fval))

        # adding complex node connections into consideration
        for n1 in self.nodes:
            conns = collections.defaultdict(lambda: set([]))
            for v, n2 in self.nodes[n1].vals.items():
                conns[n2].add(v)
            self.nodes[n1].vals = {frozenset(v): n2 for n2, v in conns.items()}

        # simplifying the features and their domains
        self.feats = sorted(self.feats)
        self.feids = {f: i for i, f in enumerate(self.feats)}
        self.fdoms = {f: sorted(self.fdoms[f]) for f in self.fdoms}

        # here we assume all features are present in the tree
        # if not, this value will be rewritten by self.parse_mapping()
        self.nof_feats = len(self.feats)

        #self.paths = collections.defaultdict(lambda: [])
        #self.extract_paths(root=self.root_node, prefix=[])
        
        

    def parse_mapping(self, mapfile):
        """
            Parse feature-value mapping from a file.
        """

        self.fvmap = {}

        with open(mapfile, 'r') as fp:
            lines = fp.readlines()

        if lines[0].startswith('OHE'):
            for i in range(int(lines[1])):
                feats = lines[i + 2].strip().split(',')
                orig, ohe = feats[0], tuple(feats[1:])
                self.ohmap.dir[orig] = tuple(ohe)
                for f in ohe:
                    self.ohmap.opp[f] = orig

            lines = lines[(int(lines[1]) + 2):]

        elif lines[0].startswith('Categorical'):
            # skipping the first comment line if necessary
            lines = lines[1:]

        # number of features
        self.nof_feats = int(lines[0].strip())
        self.feids = {}

        for line in lines[1:]:
            feat, val, real = line.strip().split()
            self.fvmap[tuple([feat, int(val)])] = '{0}{1}'.format(feat, real)
            if feat not in self.feids:
                self.feids[feat] = len(self.feids)

        assert len(self.feids) == self.nof_feats

    def convert_to_multiedges(self):
        """
            Convert ITI trees with '!=' edges to multi-edges.
        """

        # new feature domains
        fdoms = collections.defaultdict(lambda: [])

        # tentative mapping relating negative and positive values
        nemap = collections.defaultdict(lambda: collections.defaultdict(lambda: [None, None]))

        for fv, tval in self.fvmap.items():
            if '!=' in tval:
                nemap[fv[0]][tval.split('=')[1]][0] = fv[1]
            else:
                fdoms[fv[0]].append(fv[1])
                nemap[fv[0]][tval.split('=')[1]][1] = fv[1]

        # a mapping from negative values to sets
        fnmap = collections.defaultdict(lambda: {})
        for f in nemap:
            for t, vals in nemap[f].items():
                if vals[0] != None:
                    fnmap[(f, frozenset({vals[0]}))] = frozenset(set(fdoms[f]).difference({vals[1]}))

        # updating node connections
        for n in self.nodes:
            vals = {}
            for v in self.nodes[n].vals.keys():
                fn = (self.nodes[n].feat, v)
                if fn in fnmap:
                    vals[fnmap[fn]] = self.nodes[n].vals[v]
                else:
                    vals[v] = self.nodes[n].vals[v]
            self.nodes[n].vals = vals

        # updating the domains
        self.fdoms = fdoms

        # extracting the paths again
        self.paths = collections.defaultdict(lambda: [])
        self.extract_paths(root=self.root_node, prefix=[])
        
       
        

    def extract_paths(self, root, prefix):
        """
            Traverse the tree and extract explicit paths.
        """

        if root in self.terms:
            # store the path
            term = self.terms[root]
            self.paths[term].append(prefix)
        else:
            # select next node
            feat, vals = self.nodes[root].feat, self.nodes[root].vals
            for val in vals:
                self.extract_paths(vals[val], prefix + [tuple([feat, val])])

    def execute(self, inst, pathlits=False):
        """
            Run the tree and obtain the prediction given an input instance.
        """

        root = self.root_node
        depth = 0
        path = []

        # this array is needed if we focus on the path's literals only
        visited = [False for f in inst]

        while not root in self.terms:
            path.append(root)
            feat, vals = self.nodes[root].feat, self.nodes[root].vals
            visited[self.feids[feat]] = True
            tval = inst[self.feids[feat]][1]
            ###############
            # assert(len(vals) == 2)
            next_node = root
            neq = None
            for vs, dest in vals.items():
                if tval in vs:
                    next_node = dest
                    break
                else:
                    for v in vs:
                        if '!=' in self.fvmap[(feat, v)]:
                            neq = dest
                            break
            else:
                next_node = neq

            assert (next_node != root)
            ###############
            root = next_node
            depth += 1

        if pathlits:
            # filtering out non-visited literals
            for i, v in enumerate(visited):
                if not v:
                    inst[i] = None

        return path, self.terms[root], depth

    
    def execute2(self, inst):
        inst = list(map(lambda i : tuple(['f{0}'.format(i[0]), int(i[1])]), [(i, j) for i,j in enumerate(inst)]))
        return self.execute(inst)
        
    def predict(self, samples):  
        assert(samples.ndim == 2)
        pred = []
        for inst in samples:
            # input  'value1,value2,...'
            instance = list(map(lambda i : tuple(['f{0}'.format(i[0]), int(i[1])]), [(i, j) for i,j in enumerate(inst)]))  
            _, y_pred, _ = self.execute(instance)
            pred.append(y_pred)
        return np.array(pred)
        #return np.array(pred, dtype=np.float32)
    

    def execute_path(self, path):
        """
            Execute a path to obtain a class label.
        """

        root = self.root_node

        for fv in path:
            feat, vals = self.nodes[root].feat, self.nodes[root].vals
            root = vals[fv[1]]

        assert root in self.terms, 'The path does not lead to a leaf'
        return self.terms[root]
    
    
    def precision(self, inst, expl):
        """
        assessing precision of explanation: expl
        """
        #print(inst)
        inst = list(map(lambda i : tuple(['f{0}'.format(i[0]), int(i[1])]), [(i, j) for i,j in enumerate(inst)]))
        if self.verbose:
            print('inst:',inst)
            print()
        _, y_pred, _ = self.execute(inst)
        univ = {}
        for i in range(self.nof_feats):
            univ[f"f{i}"] = True
        for i in expl:
            univ[f"f{i}"] = False

        prob = 0.0 
        for path in self.paths[y_pred]:
            consistent = True
            for nd in path:
                if (not univ[nd[0]]) and (inst[int(nd[0][1:])][1] not in nd[1]):
                    consistent = False
                    break
            if consistent:
                p = dict()
                for nd in path:
                    if nd[0] not in p:
                        p[nd[0]] = nd[1]
                    else:
                        p[nd[0]] = frozenset.intersection(p[nd[0]], nd[1])
                        #assert(len(p[nd[0]]))
                d = math.prod([len(d) for f,d in p.items() if univ[f]])
                doms = math.prod([len(self.fdoms[f]) for f,d in p.items() if univ[f]])
                prob += d / doms
        
        return prob
    

    def encodeMaxSMT(self, inst, delta=1.0, verbose=0, encod="prod"):
        assert (delta >0.0 and delta <=1.0)
        inst = list(map(lambda i : tuple(['f{0}'.format(i[0]), int(i[1])]), [(i, j) for i,j in enumerate(inst)]))
        decision_path, y_pred, depth = self.execute(inst)
        pathFeats = [self.nodes[i].feat for i in decision_path]
        if verbose:
            print("path len:", depth)
        hard_f = []
        csum = []
        isum = []        
        fvars = Bools(' '.join([f's{i}' for i in range(self.nof_feats)]))
        j = 0
        for term in self.paths:
            for path in self.paths[term]:
                #pvar = Bool(f't{j}')
                j += 1
                pathlits = dict()
                for nd in path:
                    if nd[0] not in pathlits:
                        pathlits[nd[0]] = nd[1]
                    else:
                        pathlits[nd[0]] = frozenset.intersection(pathlits[nd[0]], nd[1])
                        assert(len(pathlits[nd[0]]))
                
                
                if encod == "prod":
                    mcount = []
                    for f in self.feats:
                        if (f not in pathlits):
                            mcount.append(If(Bool(f's{f[1:]}'), 1, IntVal(len(self.fdoms[f]))))
                    for f, vals in pathlits.items():
                        if inst[int(f[1:])][1] not in vals:
                            mcount.append(If(Bool(f's{f[1:]}'), 0, IntVal(len(vals))))
                        else:
                            mcount.append(If(Bool(f's{f[1:]}'), 1, IntVal(len(vals))))               
                    mcount = Product(mcount)
                else:
                    assert (encod == "sum")
                    mcount = [1]
                    for f in self.feats:
                        mcount.append(0)
                        for v in self.fdoms[f]:
                            if f not in pathlits:
                                if v == inst[int(f[1:])][1]:                               
                                    mcount[-1] += mcount[-2]
                                else:
                                    mcount[-1] += If(Bool(f's{f[1:]}'), 0, mcount[-2]) 
                            elif (v in pathlits[f]) :
                                if (v == inst[int(f[1:])][1]):
                                    mcount[-1] += mcount[-2]
                                else: 
                                    mcount[-1] += If(Bool(f's{f[1:]}'), 0, mcount[-2]) 
                    #                
                    mcount = mcount[-1]              
                
                if term == y_pred:
                    csum.append(mcount)
                else:    
                    isum.append(mcount)
                
        
        #hard_f.append(100*Sum(csum) >= (IntVal(100*delta)*Sum(csum+isum)))
        if delta == 1.0:
            hard_f.append(Sum(isum) == 0)
            ##hard_f.append(Sum(csum) == 1) 
        else:    
            hard_f.append(IntVal(100 - 100*delta) * Sum(csum) >= (IntVal(100*delta)*Sum(isum)))
        #soft_f = [Not(v) for v in fvars]
        soft_f = []
        for (v, f) in zip(fvars, [f'f{i}' for i in range(self.nof_feats)]):
            if f in pathFeats:
                soft_f.append(Not(v))
            else:
                hard_f.append(Not(v))
                
        if verbose>1:
            for i in hard_f:
                print(i)   
        
        return hard_f, soft_f

 
    def weak_rset(self, inst, delta=1.0, verbose=0):
        assert (delta >0.0 and delta <=1.0)
        #inst = list(map(lambda i : tuple([f'f{i[0]}', int(i[1])]), [(i, j) for i,j in enumerate(inst)]))
        decision_path, y_pred, depth = self.execute([(f'f{i}', int(j)) for i,j in enumerate(inst)])
        if verbose:
            print("path len:", depth)   
        expl = list(set({int(self.nodes[i].feat[1:]) for i in decision_path}))
        pathFeats = []
        for i, f in enumerate(expl):
            p  = self.precision(inst, expl[:i]+expl[i+1:])
            pathFeats.append(tuple([f, 1.0 - p]))
        pathFeats = sorted(pathFeats, key=lambda x: x[1])
        #print(pathFeats)
        expl = [j for j,_ in pathFeats]
        i = 0
        while i<len(expl):
            prec = self.precision(inst, expl[:i]+expl[i+1:])
            #print(prec)
            if prec >= delta:
                expl = expl[:i]+expl[i+1:]
            else:
                i = i + 1
        return expl  
    
    def isDRset(self, inst, expl, delta, verbose=0):
        """
            Check whether the given expl is minimal-subset
        """
        formula, soft_f = self.encodeMaxSMT(inst, delta) 
        #n = len(formula)
        cl = []
        for p in soft_f:
            assert(str(p)[4:-1].startswith('s'))
            if str(p)[5:-1] in expl:
                cl.append(p)
            else:
                formula.append(p)
        cl = Or(cl) 
        formula.append(cl)
        slv = Solver()
        for c in formula[:-1]:
            slv.add(c)
        if(slv.check() == unsat): # expl is not WeakPAXp
            return False
        #print(formula[n:])
        slv.add(formula[-1])
        return (slv.check() == unsat)
        
#
#==============================================================================
def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'hi:d:o:m:t:x:ve:w',
                                   ['help',
                                    'inst=',
                                    'delta=',
                                    'output=',
                                    'map=',
                                    'tree=',
                                    'expl=',
                                    'verb',
                                    'enc=',
                                    'weak'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    # init 
    verb = 0
    mapfile = None
    treefile = None
    log = ''
    inst = ''
    expl = None
    delta = None
    enc = 'prod'
    weak = False

    for opt, arg in opts:
        if  opt in ('-m', '--map'):
            mapfile = str(arg)  
        elif  opt in ('-t', '--tree'):
            treefile = str(arg)
        elif opt in ('-i', '--inst'):  
            inst = str(arg).strip().split(',')
        elif opt in ('-d', '--delta'):
            delta = float(str(arg))            
        elif opt in ('-x', '--expl'):  
            expl = str(arg).strip().split(',')            
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-v', '--verb'):
            verb += 1
        elif opt in ('-o', '--output'):
            log = str(arg)
        elif opt in ('-e', '--enc'):
            enc = str(arg)
        elif opt in ('-w', '--weak'):
            weak = True
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)


    return log, verb, mapfile, treefile, inst, expl, delta, enc, weak, args

#
#==============================================================================
def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] decision-tree')
    print('Options:')
    print('        -d, --delta              Probability threshold')
    print('                                 Available values: [0.0 .. 1.0], AXp (default = 1.0)')
    print('        -e, --enc=<string>       SMT encoding to use')
    print('                                 Available values: prod, sum (default = prod)')
    print('        -h, --help')
    print('        -i, --inst=<csv>         Input instance to explain')
    print('                                 (comma-separated values)')
    
    print('        -m, --map=<string>       Path to mapping file for correct feature-value handling')
    print('        -t, --tree=<string>      Path to dtree (.dt) file')
    print('        -v, --verb               Be verbose (show comments)')
    print('        -w, --weak               Compute ApproxPAXp explanation')
    print('        -x, --expl=<csv>         Explanation to measure its precision or check whether it is minimal-subset PAXp')


#
#==============================================================================
if __name__ == '__main__':
    # parsing command-line options
    log, verb, mapfile, treefile, inst, expl, delta, enc, weak_expl, files = parse_options()

    if not (treefile and mapfile):
        print('dtree file or map file is missing!')
        exit()
        
    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime  
        
    dtree = DecisionTree(treefile, mapfile=mapfile, medges=True, verbose=verb)
    
    if (expl and delta):
        print(f'Deciding if ({",".join(expl)}) is a DRSet:')
        print( dtree.isDRset(inst, expl, delta) )
        exit()
    
    if expl:
        pre = dtree.precision(inst, expl)
        print("Precision: {0:.2f}".format(pre*100))
        
    elif weak_expl and delta:
        expl = dtree.weak_rset(inst, delta)
        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer        
        expl = sorted(expl)
        
        print('expl features:', ','.join([f'f{i}' for i in expl]))     
        print('expl feature ids:', ','.join([str(i+1) for i in expl]))
        print('expl literals:', ','.join([f'f{i}={inst[i]}' for i in expl]))
        print('expl len:', len(expl))

        pre = dtree.precision(inst, expl)
        print(f"Precision: {100*pre:.2f}")
        
        print(f"time: {timer:.2f}")
        print()
        
    elif delta :        
        hard_f, soft_f = dtree.encodeMaxSMT(inst, delta, verb, enc)
        mxs = Optimize()
        for i,c in enumerate(hard_f):
            mxs.add(c) # hard constraint
            mxs.assert_and_track(c, f'c{i}')    
        for c in soft_f:
            mxs.add_soft(c) # soft constraint
        
        #if mxs.check() != sat:
        #    print(mxs.check())
        #    print(mxs.unsat_core())
        #    for c in mxs.unsat_core()[:-1]:
        #        print(hard_f[int(str(c)[1:])] )
        #    print(mxs.statistics())
        #    exit()
        #print("soft clauses:", len(soft_f))
        #exit()    
        assert(mxs.check() == sat)
        if verb:
            print(mxs.statistics())
            #print(mxs.model()) 
        
            
        model = mxs.model()
        fvars = [v for v in model.decls() if (str(v)[0]=='s')]
        expl = [int(str(v)[1:]) for v in fvars if (model[v]==BoolVal(True))]
        expl = sorted(expl)
        
        print('expl features:', ','.join([f'f{i}' for i in expl]))     
        print('expl feature ids:', ','.join([str(i+1) for i in expl]))
        print('expl literals:', ','.join([f'f{i}={inst[i]}' for i in expl]))
        print('expl len:', len(expl))

        pre = dtree.precision(inst, expl)
        print(f"Precision: {100*pre:.2f}")
        
        print(f"time: {timer:.2f}")
        print()        
    
