"""END-TO-END: in the FINISHED crop, where does the subject actually sit?

The unit test proves _place_frac returns 0.5. It does NOT prove the rendered
window is centred on the subject — easing, clamping and stabilize_box all sit
between the two. This measures the real thing: for every detection frame,
the chosen target box vs the crop rect that frame produced.
"""
import json, collections
import reframe_v2 as r, main as m, subject_policy as sp

META = ("/app/output/adcd8a49-28c5-4c82-b806-2b08c7e63bbb/"
        "Ep_115_Pop_The_Balloon_Or_Find_Love__With_Arlette_Amuli_metadata.json")
t = json.load(open(META))["transcript"]
transcript = t if isinstance(t, dict) else {"segments": t}

last = {"box": None}
orig_decide = sp.SubjectPolicy.decide
def decide(self, cands, ev, fn, frame_width=None):
    box, cid, tier = orig_decide(self, cands, ev, fn, frame_width=frame_width)
    if box is not None:
        last["box"] = tuple(box)
    return box, cid, tier
sp.SubjectPolicy.decide = decide

samples = []
orig_crop = m.SmoothedCameraman.get_crop_box
def get_crop_box(self, force_snap=False):
    x1, y1, x2, y2 = orig_crop(self, force_snap=force_snap)
    b = last["box"]
    if b is not None and x2 > x1:
        subj_cx = b[0] + b[2] / 2.0
        samples.append(((subj_cx - x1) / (x2 - x1), subj_cx, x1, x2))
    return x1, y1, x2, y2
m.SmoothedCameraman.get_crop_box = get_crop_box

r.render("/app/scratch_ptb/ptb_raw.mp4", "/app/scratch_ptb/_centre.mp4", 0.75,
         transcript=transcript, clip_start=481.0, clip_end=513.0)

pos = [s[0] for s in samples]
hist = collections.Counter(min(9, max(0, int(p * 10))) for p in pos)
print(f"\nCENTRECHECK frames={len(pos)}")
print("where the subject sits inside the rendered 3:4 crop:")
for b in range(10):
    n = hist.get(b, 0)
    print(f"  {b/10:.1f}-{(b+1)/10:.1f} {'#' * int(60*n/max(1,len(pos))):<60} {n}")
centred = sum(1 for p in pos if abs(p - 0.5) < 0.10)
bad = sum(1 for p in pos if p < 0.2 or p > 0.8)
print(f"\ncentred (within 10% of middle): {100.0*centred/max(1,len(pos)):.0f}%")
print(f"near the crop EDGE (<0.2 or >0.8): {100.0*bad/max(1,len(pos)):.0f}%")
worst = sorted(samples, key=lambda s: abs(s[0] - 0.5), reverse=True)[:5]
print("\nworst frames (pos, subject_cx, crop x1..x2):")
for p, cx, x1, x2 in worst:
    print(f"  pos={p:.2f}  subj={cx:.0f}  crop={x1:.0f}..{x2:.0f}")
