# research/

Research dossier for the LeapXELA cube-reorientation plateau.

| Path | What it is |
|---|---|
| `RESEARCH.md` | **Start here.** The synthesis: ranked hypotheses, a comparison table against every working reference implementation, diagnostics to run before training, and the indexed source list. |
| `CONTEXT.md` | The failure brief every analysis was written against (env config, PPO config, observed curves). |
| `analysis/` | Twelve deep analyses, one per source cluster, same section structure throughout. |
| `tools/` | Scripts that (re)build `sources/`. |
| `sources/` | **Not committed** — see below. |

## Regenerating `sources/`

`research/sources/` holds ~2.4 MB of third-party material: 19 arXiv papers converted
to text, four reference repos bundled to one file each, four GitHub issue/discussion
threads, and MuJoCo/MJX/MJWarp documentation pages. It is gitignored rather than
committed, because it is other people's copyrighted text and source code and it is
reproducible from scratch.

To rebuild it (needs `beautifulsoup4`, `git`, and `gh` authenticated):

```bash
mkdir -p research/sources
python3 research/tools/fetch_arxiv_sources.py     # 16 core papers
python3 research/tools/fetch_extra_papers.py      # 3 later additions
python3 research/tools/fetch_threads_and_docs.py  # GitHub threads + MuJoCo docs

git clone --depth 1 https://github.com/google-deepmind/mujoco_playground.git /tmp/mpg
git clone --depth 1 https://github.com/Msornerrrr/in-hand-rotation-mjlab.git   /tmp/mjlab_ih
git clone --depth 1 https://github.com/HaozhiQi/hora.git                       /tmp/hora_repo
git clone --depth 1 https://github.com/leap-hand/LEAP_Hand_Sim.git             /tmp/leaphandsim
python3 research/tools/bundle_repos.py
```

The analyses cite sources by filename, so they stay readable without `sources/` present.

`tools/fetch_notion_page.py` + `tools/render_notion_blocks.py` dump the "Training
LeapXELA" Notion log to markdown — Notion is client-rendered, so a plain HTTP fetch
returns an empty shell and the block API has to be walked instead.
