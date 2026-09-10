import os
import subprocess, os, re, json
from bs4 import BeautifulSoup
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sources")
os.makedirs(OUT, exist_ok=True)
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"

def to_text(html):
    soup=BeautifulSoup(html,"html.parser")
    for t in soup(["script","style","noscript","svg","header","footer","nav"]): t.decompose()
    txt=soup.get_text("\n"); txt=re.sub(r"\n{3,}","\n\n",txt); txt=re.sub(r"[ \t]{2,}"," ",txt)
    return txt.strip()

def fetch(url):
    return subprocess.run(["curl","-sL","--max-time","90","-A",UA,url],capture_output=True).stdout

def save(name,text):
    open(os.path.join(OUT,name),"w",encoding="utf-8").write(text)
    print(f"{len(text):>9,}  {name}")

# --- GitHub discussions / issues via GraphQL+REST (authenticated) ---
def gh_discussion(owner,repo,num,name,url):
    q='''query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){discussion(number:$n){
      title bodyText comments(first:100){nodes{author{login} bodyText replies(first:50){nodes{author{login} bodyText}}}}}}}'''
    r=subprocess.run(["gh","api","graphql","-f",f"query={q}","-F",f"o={owner}","-F",f"r={repo}","-F",f"n={num}"],
                     capture_output=True,text=True)
    try: d=json.loads(r.stdout)["data"]["repository"]["discussion"]
    except Exception as e:
        print(f"  !! discussion {owner}/{repo}#{num} failed: {r.stderr[:200]}"); return
    out=[f"SOURCE: {url}\nTITLE: {d['title']}\n\n{d['bodyText']}"]
    for c in d["comments"]["nodes"]:
        out.append(f"\n--- COMMENT by {(c.get('author') or {}).get('login','?')} ---\n{c['bodyText']}")
        for rp in c["replies"]["nodes"]:
            out.append(f"\n  --- REPLY by {(rp.get('author') or {}).get('login','?')} ---\n{rp['bodyText']}")
    save(name,"\n".join(out))

def gh_issue(owner,repo,num,name,url):
    r=subprocess.run(["gh","issue","view",str(num),"-R",f"{owner}/{repo}","--comments",
                      "--json","title,body,comments"],capture_output=True,text=True)
    try: d=json.loads(r.stdout)
    except Exception: print(f"  !! issue {num} failed"); return
    out=[f"SOURCE: {url}\nTITLE: {d['title']}\n\n{d['body']}"]
    for c in d.get("comments",[]):
        out.append(f"\n--- COMMENT by {c.get('author',{}).get('login','?')} ---\n{c['body']}")
    save(name,"\n".join(out))

gh_discussion("mujocolab","mjlab",656,"gh_mjlab_discussion_656_leap_cube_sim2real.txt",
              "https://github.com/mujocolab/mjlab/discussions/656")
gh_discussion("google-deepmind","mujoco",2915,"gh_mujoco_disc_2915_nefc_ncon_overflow.txt",
              "https://github.com/google-deepmind/mujoco/discussions/2915")
gh_discussion("google-deepmind","mujoco_playground",197,"gh_playground_disc_197_mujoco_warp.txt",
              "https://github.com/google-deepmind/mujoco_playground/discussions/197")
gh_issue("google-deepmind","mujoco_playground",302,"gh_playground_issue_302_leap_kp_kd.txt",
         "https://github.com/google-deepmind/mujoco_playground/issues/302")

# --- Simulation docs ---
DOCS=[("https://mujoco.readthedocs.io/en/latest/mjwarp/","docs_mujoco_warp.txt"),
      ("https://mujoco.readthedocs.io/en/stable/mjx.html","docs_mjx.txt"),
      ("https://mujoco.readthedocs.io/en/stable/modeling.html","docs_mujoco_modeling.txt"),
      ("https://mujoco.readthedocs.io/en/stable/computation/index.html","docs_mujoco_computation_contacts.txt")]
for url,name in DOCS:
    h=fetch(url)
    if h and len(h)>3000: save(name, f"SOURCE URL: {url}\n\n{to_text(h)}")
    else: print(f"  !! FAILED {name}")
