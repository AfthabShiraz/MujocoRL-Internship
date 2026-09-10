import json, subprocess

PAGE="31a90c5e-820d-80ba-8b17-c5badfeb0eeb"
SPACE="2336db19-d98b-47e8-826c-eed09ae2f483"
blocks={}

def post(url,payload):
    out=subprocess.run(["curl","-s","-X","POST",url,
        "-H","Content-Type: application/json","-H","User-Agent: Mozilla/5.0","-d",json.dumps(payload)],
        capture_output=True,text=True).stdout
    try: return json.loads(out)
    except: return {}

def absorb(rm):
    for bid,b in (rm.get("block") or {}).items():
        v=b.get("value",{})
        v=v.get("value",v)
        if v: blocks[bid]=v

cursor={"stack":[]}; chunk=0
while True:
    d=post("https://www.notion.so/api/v3/loadPageChunk",
        {"pageId":PAGE,"limit":100,"cursor":cursor,"chunkNumber":chunk,"verticalColumns":False})
    absorb(d.get("recordMap",{}))
    cur=d.get("cursor",{})
    if not cur.get("stack"): break
    cursor=cur; chunk+=1
    if chunk>50: break

def missing():
    need=set()
    for v in list(blocks.values()):
        for c in (v.get("content") or []):
            if c not in blocks: need.add(c)
    return need

for _ in range(15):
    need=missing()
    if not need: break
    need=list(need)
    for i in range(0,len(need),80):
        batch=need[i:i+80]
        d=post("https://www.notion.so/api/v3/syncRecordValues",
          {"requests":[{"pointer":{"table":"block","id":b,"spaceId":SPACE},"version":-1} for b in batch]})
        absorb(d.get("recordMap",{}))
        if not d.get("recordMap"):
            absorb({"block":d.get("recordMapWithRoles",{}).get("block",{})})

print("BLOCKS",len(blocks),"MISSING",len(missing()))
json.dump(blocks,open("/tmp/blocks.json","w"))
