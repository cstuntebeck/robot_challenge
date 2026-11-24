import re,sys,statistics as st
# Mittel der letzten 30 Iterationen je Log (Claas, 23.11.)
keys={"sturz":r"Episode_Termination/bad_orientation:\s*([\d.eE+-]+)","len":r"Mean episode length:\s*([\d.]+)","v":r"error_vel_xy:\s*([\d.]+)","w":r"error_vel_yaw:\s*([\d.]+)","it":r"Learning iteration (\d+)"}
for a in sys.argv[1:]:
    name,f=a.split("=")
    try: t=open(f,errors="ignore").read()
    except Exception as e: print(name,"FEHLT"); continue
    blocks=t.split("Learning iteration")[1:]
    rows=[]
    for b in blocks:
        b="Learning iteration"+b
        r={}
        for k,p in keys.items():
            m=re.search(p,b); r[k]=float(m.group(1)) if m else None
        rows.append(r)
    last=rows[-30:]
    def mean(k):
        v=[r[k] for r in last if r[k] is not None]; return st.mean(v) if v else float("nan")
    print(f"{name}: n={len(rows)} sturz={mean('sturz')*100:.2f}% len={mean('len'):.0f} v={mean('v'):.3f} w={mean('w'):.3f} done={'TRAIN_DONE' in t}")
