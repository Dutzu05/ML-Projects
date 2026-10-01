"""
CCC Level 3 - classify each flock's species from its BOP routes

Usage:  python level3.py <data_dir> <out_dir>
"""
import glob, os, sys
from features import load, load_temps, features

def classify(F):
    out={}
    same=F[F.identical]                       # whole flock flies one route
    for fid,r in same.iterrows():             # predator: nodes ⊂ another route
        if any(r.nodes < o.nodes for j,o in same.iterrows() if j!=fid):
            out[fid]='Sticky Wolfthroat'
    for fid,r in same.iterrows():
        if fid in out: continue
        out[fid]='Medieval Bluetit' if r.palin else 'Flanking Blackfinch'
    rest=F[~F.identical]
    if rest.empty: return dict(sorted(out.items()))
    out[rest.prefix.idxmax()]='Rusty Goldhammer'     # long shared prefix, then split
    rest=rest.drop(rest.prefix.idxmax())
    out[rest.mean_temp.idxmax()]='Hurracurra Bird'   # hottest BOPs
    rest=rest.drop(rest.mean_temp.idxmax())
    if not rest.empty:
        out[rest.index[0] if len(rest)==1 else rest.mean_temp.idxmin()]='Red Firefinch'
    assert len(set(out.values()))==len(out), out
    return dict(sorted(out.items()))

data_dir, out_dir = (sys.argv[1:3] + ['.', '.'])[:2]
os.makedirs(out_dir, exist_ok=True)
temps=load_temps(data_dir)
for f in sorted(glob.glob(os.path.join(data_dir, 'in_level-3_*.txt'))):
    name=f.split('in_level-3_')[1]
    res=classify(features(load(f), temps))
    txt='Flock ID,Species\n'+''.join(f'{k},{v}\n' for k,v in res.items())
    if 'example' in name:
        exp=open(os.path.join(data_dir, 'out_level-3_0-example.txt')).read()
        print('example matches:',txt.strip()==exp.strip()); continue
    open(os.path.join(out_dir, f'out_level-3_{name}'),'w').write(txt)
    print(name); print(txt)
