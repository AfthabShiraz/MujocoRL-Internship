import json
blocks=json.load(open("/tmp/blocks.json"))
PAGE="31a90c5e-820d-80ba-8b17-c5badfeb0eeb"
def txt(prop):
    if not prop: return ""
    s=""
    for item in prop:
        t=item[0]
        for f in (item[1] if len(item)>1 else []) or []:
            if f[0]=="e": t="$"+(f[1] if len(f)>1 else "")+"$"
        s+=t
    return s
out=[]
def render(bid,depth=0):
    b=blocks.get(bid)
    if not b: return
    t=b.get("type"); p=b.get("properties") or {}
    title=txt(p.get("title")); ind="  "*depth
    if t=="page": out.append(f"# {title}")
    elif t=="header": out.append(f"\n## {title}")
    elif t=="sub_header": out.append(f"\n### {title}")
    elif t=="sub_sub_header": out.append(f"\n#### {title}")
    elif t=="text": out.append(ind+title)
    elif t=="bulleted_list": out.append(ind+"- "+title)
    elif t=="numbered_list": out.append(ind+"1. "+title)
    elif t=="to_do": out.append(ind+f"- [{'x' if txt(p.get('checked'))=='Yes' else ' '}] "+title)
    elif t=="toggle": out.append(ind+"▸ "+title)
    elif t=="code": out.append(ind+f"```{txt(p.get('language'))}\n{title}\n```")
    elif t=="quote": out.append(ind+"> "+title)
    elif t=="callout": out.append(ind+"[!] "+title)
    elif t=="divider": out.append(ind+"---")
    elif t=="image": out.append(ind+f"[IMAGE] {txt(p.get('caption'))}")
    elif t=="video": out.append(ind+"[VIDEO]")
    elif t=="equation": out.append(ind+"$$"+title+"$$")
    elif t=="table_row": out.append(ind+"| "+" | ".join(txt(v) for v in p.values())+" |")
    elif t=="table": out.append(ind+"[TABLE]")
    else: out.append(ind+f"[{t}] {title}")
    nd = depth if t in ("page","header","sub_header","sub_sub_header") else depth+1
    for c in b.get("content") or []: render(c,nd)
render(PAGE)
print("\n".join(out))
