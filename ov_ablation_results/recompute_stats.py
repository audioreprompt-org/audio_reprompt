import glob, re, itertools, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
S="data/ablations/scores/"; R="data/ablations/reprompts/"; A="data/ablations/analysis/"
RUNS={"B1":"R20261107_151324","B2":"R20261116_234152","B3":"R20261117_021911"}
rng=np.random.default_rng(42)
def load(exp):
    run=RUNS[exp]; out={}
    base_cache_path = S + "clap_score_results_raw_prompts_baseline.csv"
    base_cache = pd.read_csv(base_cache_path)[["id_prompt","clap_score"]].rename(columns={"clap_score":"raw_true"}) if glob.glob(base_cache_path) else None
    for f in sorted(glob.glob(S+f"clap_score_results_reprompt_outputs_*_{exp}[a-g]_*{run}.csv")):
        v=re.search(rf"_({exp}[a-g])_",f).group(1)
        rep=pd.read_csv(f); raw=pd.read_csv(f.replace("reprompt_outputs_","prompt_outputs_").replace(".csv","_cross.csv"))
        tf=glob.glob(R+f"pipeline_results_*_{v}_*{run}.csv")[0]; t=pd.read_csv(tf)[["id_prompt","taste"]]
        assert rep.id_prompt.is_unique and raw.id_prompt.is_unique
        d=rep[["id_prompt","clap_score"]].rename(columns={"clap_score":"rep"}).merge(raw[["id_prompt","clap_score"]].rename(columns={"clap_score":"raw"}),on="id_prompt").merge(t,on="id_prompt")
        if base_cache is not None:
            d=d.merge(base_cache, on="id_prompt", how="left")
        print(exp,v,"rows rep/raw/merged",len(rep),len(raw),len(d),"NaN",int(d.isna().sum().sum()))
        out[v]=d.set_index("id_prompt")
    return out
def boot(x,B=10000):
    idx=rng.integers(0,len(x),(B,len(x))); m=x.values[idx].mean(1); return np.percentile(m,[2.5,97.5])
def paired(a,b):
    d=(a-b).dropna(); n=len(d)
    t,p=stats.ttest_rel(a.loc[d.index],b.loc[d.index])
    try: w,pw=stats.wilcoxon(d)
    except Exception: w,pw=np.nan,np.nan
    sh=stats.shapiro(d).pvalue
    dz=d.mean()/d.std(ddof=1); dav=d.mean()/np.sqrt((a.var()+b.var())/2)
    lo,hi=boot(d); 
    return dict(n=n,mean_a=a.mean(),mean_b=b.mean(),diff=d.mean(),t=t,p_t=p,W=w,p_w=pw,shapiro_p=sh,dz=dz,dav=dav,ci_lo=lo,ci_hi=hi,prop_pos=(d>0).mean(),skew=stats.skew(d))
def holm(ps): return multipletests(ps,method="holm")[1]
def run(exp):
    D=load(exp); vs=list(D); print("\n=====",exp,vs)
    rows=[dict(variant=v,**paired(D[v].rep,D[v].raw)) for v in vs]
    T=pd.DataFrame(rows); T["p_t_holm"]=holm(T.p_t); T["p_w_holm"]=holm(T.p_w)
    pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
    print("-- A) reprompt vs raw (within variant)\n",T.round(4).to_string(index=False))
    # per taste
    rows=[]
    for v in vs:
        for tz,g in D[v].groupby("taste"): rows.append(dict(variant=v,taste=tz,**paired(g.rep,g.raw)))
    Tt=pd.DataFrame(rows); Tt["p_t_holm"]=holm(Tt.p_t); Tt["p_w_holm"]=holm(Tt.p_w)
    print("-- A2) per taste\n",Tt[["variant","taste","n","mean_b","mean_a","diff","p_t","p_w","p_t_holm","p_w_holm","dz"]].round(4).to_string(index=False))
    # between variants
    W=pd.DataFrame({v:D[v].rep for v in vs}).dropna(); Wc=pd.DataFrame({v:D[v].raw for v in vs}).dropna(); Wd=pd.DataFrame({v:D[v].rep-D[v].raw for v in vs}).dropna()
    print("-- B) between variants, complete cases n=",len(W))
    print("means cross (original prompt <-> reprompt audio):",Wc.mean().round(4).to_dict())
    print("means rep   (reprompt text   <-> reprompt audio):",W.mean().round(4).to_dict())
    print("Shapiro per variant (cross):",{v:round(stats.shapiro(Wc[v]).pvalue,4) for v in vs})
    print("Shapiro per variant (rep):",{v:round(stats.shapiro(W[v]).pvalue,4) for v in vs})
    if len(vs)>2:
        for name,X in (("cross",Wc),("rep",W),("diff",Wd)):
            fr=stats.friedmanchisquare(*[X[v] for v in vs]); k=len(vs)
            print(f"Friedman[{name}] chi2={fr.statistic:.3f} p={fr.pvalue:.4g} Kendall W={fr.statistic/(len(X)*(k-1)):.3f}")
            long=X.reset_index().melt(id_vars="id_prompt",var_name="variant",value_name="y")
            a=AnovaRM(long,"y","id_prompt",within=["variant"]).fit().anova_table; print(f"RM-ANOVA[{name}]\n",a.round(5).to_string())
            # Mauchly not in statsmodels; report GG epsilon manually
            Xm=X.values; k=Xm.shape[1]; C=np.cov(Xm.T); P=np.eye(k)-1/k; Cc=P@C@P
            eps=np.trace(Cc)**2/((k-1)*np.sum(Cc**2)); print("Greenhouse-Geisser eps=%.3f"%eps)
            df1=(k-1)*eps; df2=(k-1)*(len(X)-1)*eps; F=a["F Value"].iloc[0]; print("GG-adj p=%.4g"%stats.f.sf(F,df1,df2))
    pairs=list(itertools.combinations(vs,2)) if exp!="B3" else [(v,"B3b") for v in vs if v!="B3b"]
    for name,X in (("cross",Wc),("rep",W),("diff",Wd)):
        rows=[dict(pair=f"{a} vs {b}",**paired(X[a],X[b])) for a,b in pairs]
        P=pd.DataFrame(rows); P["p_t_holm"]=holm(P.p_t); P["p_w_holm"]=holm(P.p_w)
        print(f"-- C) pairwise[{name}] ({'all pairs' if exp!='B3' else 'vs B3b control'}), Holm\n",P[["pair","diff","p_t","p_w","p_t_holm","p_w_holm","dz","ci_lo","ci_hi","shapiro_p"]].round(4).to_string(index=False))
    if exp=="B3":  # planned factor contrasts
        c=lambda a,b:(W[a]-W[b])
        print("-- D) planned: creative(c) vs deterministic(a):",{k:round(v,4) for k,v in paired(W.B3c,W.B3a).items() if k in("diff","p_t","p_w","dz")})
        print("   effort high vs default-creative (d vs c):",{k:round(v,4) for k,v in paired(W.B3d,W.B3c).items() if k in("diff","p_t","p_w","dz")})
        print("   effort low vs default-determ (e vs a):",{k:round(v,4) for k,v in paired(W.B3e,W.B3a).items() if k in("diff","p_t","p_w","dz")})
        print("   none vs default (f vs b):",{k:round(v,4) for k,v in paired(W.B3f,W.B3b).items() if k in("diff","p_t","p_w","dz")})
for e in RUNS: run(e)
