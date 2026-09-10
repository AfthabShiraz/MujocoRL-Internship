import os, glob
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sources")
os.makedirs(OUT, exist_ok=True)

def bundle(name, header, files):
    parts=[header,""]
    total=0
    for f in files:
        if not os.path.isfile(f): 
            parts.append(f"\n### MISSING: {f}\n"); continue
        try: txt=open(f,encoding="utf-8",errors="replace").read()
        except Exception as e: 
            parts.append(f"\n### UNREADABLE {f}: {e}\n"); continue
        rel=f.replace("/tmp/","")
        parts.append(f"\n\n{'='*90}\n===== FILE: {rel}\n{'='*90}\n{txt}")
        total+=len(txt)
    p=os.path.join(OUT,name)
    open(p,"w",encoding="utf-8").write("\n".join(parts))
    print(f"{os.path.getsize(p):>9,}  {name}  ({len(files)} files)")

L="/tmp/mpg/mujoco_playground/_src/manipulation/leap_hand"
bundle("repo_mujoco_playground_leap_hand.txt",
  "REPO: https://github.com/google-deepmind/mujoco_playground\nPATH: mujoco_playground/_src/manipulation/leap_hand/ + config/manipulation_params.py\nThis is the REFERENCE implementation of LeapCubeReorient / LeapCubeRotateZAxis.",
  [f"{L}/README.md", f"{L}/reorient.py", f"{L}/rotate_z.py", f"{L}/base.py",
   f"{L}/leap_hand_constants.py", f"{L}/__init__.py",
   f"{L}/xmls/leap_rh_mjx.xml", f"{L}/xmls/reorientation_cube.xml", f"{L}/xmls/scene_mjx_cube.xml",
   "/tmp/mpg/mujoco_playground/config/manipulation_params.py",
   "/tmp/mpg/mujoco_playground/_src/manipulation/__init__.py",
   "/tmp/mpg/mujoco_playground/_src/mjx_env.py",
   "/tmp/mpg/mujoco_playground/_src/reward.py",
   "/tmp/mpg/learning/train_jax_ppo.py"])

M="/tmp/mjlab_ih"
mj=sorted(glob.glob(f"{M}/src/in_hand_rotation_mjlab/tasks/hand_cube/**/*.py",recursive=True))
mj+=sorted(glob.glob(f"{M}/src/in_hand_rotation_mjlab/robots/**/*.py",recursive=True))
mj+=sorted(glob.glob(f"{M}/*.md"))+sorted(glob.glob(f"{M}/docs/*.md"))
bundle("repo_mjlab_in_hand_rotation_leap.txt",
  "REPO: https://github.com/Msornerrrr/in-hand-rotation-mjlab\nDISCUSSION: https://github.com/mujocolab/mjlab/discussions/656\nA WORKING modern sim-to-real LEAP Hand in-hand cube rotation pipeline built on mjlab (MuJoCo Warp).",
  mj)

H="/tmp/hora_repo"
bundle("repo_hora_allegro_rotation.txt",
  "REPO: https://github.com/HaozhiQi/hora  (paper: arXiv 2210.04887, CoRL 2022)\nReference reward/observation/curriculum design for in-hand rotation.",
  [f"{H}/README.md", f"{H}/hora/tasks/allegro_hand_hora.py",
   f"{H}/configs/task/AllegroHandHora.yaml", f"{H}/configs/train/AllegroHandHora.yaml",
   f"{H}/configs/config.yaml", f"{H}/docs/dev.md"])

S="/tmp/leaphandsim"
bundle("repo_leap_hand_sim_official.txt",
  "REPO: https://github.com/leap-hand/LEAP_Hand_Sim\nThe OFFICIAL LEAP Hand IsaacGym sim/RL cube-rotation environment from the LEAP Hand authors.",
  [f"{S}/README.md", f"{S}/leapsim/tasks/leap_hand_rot.py",
   f"{S}/leapsim/cfg/task/LeapHandRot.yaml", f"{S}/leapsim/cfg/train/LeapHandRotPPO.yaml",
   f"{S}/leapsim/cfg/config.yaml"])
