import os
import subprocess, os, re
from bs4 import BeautifulSoup
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sources")
os.makedirs(OUT, exist_ok=True)
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
def to_text(h):
    s=BeautifulSoup(h,"html.parser")
    for t in s(["script","style","noscript","svg"]): t.decompose()
    return re.sub(r"[ \t]{2,}"," ",re.sub(r"\n{3,}","\n\n",s.get_text("\n"))).strip()
for aid,name in [("2501.05439","from_simple_to_complex_inhand_reorientation"),
                 ("2510.08556","dexndm_joint_neural_dynamics_rotation"),
                 ("2508.01695","dexremoe_reorientation_mixture_of_experts")]:
    for u in (f"https://arxiv.org/html/{aid}v2",f"https://arxiv.org/html/{aid}v1",
              f"https://arxiv.org/html/{aid}",f"https://ar5iv.labs.arxiv.org/html/{aid}"):
        h=subprocess.run(["curl","-sL","--max-time","90","-A",UA,u],capture_output=True).stdout
        if len(h)<5000: continue
        t=to_text(h)
        if len(t)<6000: continue
        p=f"{OUT}/{name}__arxiv_{aid}.txt"
        open(p,"w").write(f"SOURCE URL: {u}\narXiv: https://arxiv.org/abs/{aid}\n\n{t}")
        print(f"{len(t):>9,}  {name}"); break
    else: print(f"  !! FAILED {name}")
