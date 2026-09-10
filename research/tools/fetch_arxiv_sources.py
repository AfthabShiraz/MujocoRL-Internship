import os
import subprocess, sys, os, re
from bs4 import BeautifulSoup

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sources")
os.makedirs(OUT, exist_ok=True)
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"

def get(url):
    r=subprocess.run(["curl","-sL","--max-time","90","-A",UA,url],capture_output=True)
    return r.stdout

def to_text(html):
    soup=BeautifulSoup(html,"html.parser")
    for t in soup(["script","style","noscript","svg"]): t.decompose()
    txt=soup.get_text("\n")
    txt=re.sub(r"\n{3,}","\n\n",txt)
    txt=re.sub(r"[ \t]{2,}"," ",txt)
    return txt.strip()

def save(name, text):
    p=os.path.join(OUT,name)
    open(p,"w",encoding="utf-8").write(text)
    print(f"{len(text):>9,}  {name}")

ARXIV = [
 ("2502.08844","mujoco_playground_paper"),
 ("1808.00177","openai_dactyl_learning_dexterity"),
 ("1910.07113","openai_rubiks_cube_ADR"),
 ("2111.03043","chen_general_inhand_reorientation_corl21"),
 ("2211.11744","chen_visual_dexterity"),
 ("2210.13702","dextreme"),
 ("2210.04887","hora_rapid_motor_adaptation"),
 ("2303.10880","rotating_without_seeing_touch_dexterity"),
 ("2309.09979","qi_general_rotation_vision_touch"),
 ("2405.07391","anyrotate"),
 ("2509.14984","role_of_touch_taxel_distribution"),
 ("2509.07445","text2touch"),
 ("2312.01853","robot_synesthesia"),
 ("2309.06440","leap_hand_rss2023"),
 ("2605.28812","beyond_binary_uskin_contact_rep"),
 ("2603.04531","ptld_tactile_latent_distillation"),
]

for aid,name in ARXIV:
    best=None
    for url in (f"https://arxiv.org/html/{aid}v3", f"https://arxiv.org/html/{aid}v2",
                f"https://arxiv.org/html/{aid}v1", f"https://arxiv.org/html/{aid}",
                f"https://ar5iv.labs.arxiv.org/html/{aid}"):
        h=get(url)
        if not h or len(h)<5000: continue
        t=to_text(h)
        if "not been rendered" in t[:2000] or len(t)<6000: continue
        best=(url,t); break
    if best:
        save(f"{name}__arxiv_{aid}.txt", f"SOURCE URL: {best[0]}\narXiv: https://arxiv.org/abs/{aid}\n\n{best[1]}")
    else:
        print(f"  !! FAILED html: {name} ({aid})")
