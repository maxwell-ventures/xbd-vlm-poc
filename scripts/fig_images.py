#!/usr/bin/env python3
"""Generate the image-composite figures (PIL) into outputs/figures/.

Uses the real chips (data/chips) and full tiles (data/raw), joined to model
generations in outputs/eval/*.jsonl by building uid.

    .venv/bin/python scripts/fig_images.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbd_vlm.schema import parse_response

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, SURFACE, LINE = (34,32,29), (107,102,94), (252,252,251), (231,227,218)
SEV = {"no-damage":(233,196,106), "minor-damage":(224,139,63),
       "major-damage":(200,90,60), "destroyed":(127,43,43)}
OK, BAD = (46,139,107), (181,72,47)

def font(sz, bold=False):
    for p in ([ "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
                "/System/Library/Fonts/Supplemental/Arial.ttf",
               "/System/Library/Fonts/Helvetica.ttc"]):
        try: return ImageFont.truetype(p, sz)
        except OSError: continue
    return ImageFont.load_default()

F = {k: font(s, b) for k, (s, b) in {
    "h":(30,True),"sub":(18,False),"lab":(19,True),"cap":(16,False),"mono":(16,False),
    "tag":(15,True),"sm":(14,False)}.items()}

def chip(uid, disaster, phase="post"):
    p = Path(f"data/chips/{disaster}/{uid}_{phase}.png")
    return Image.open(p).convert("RGB") if p.exists() else None

def load_preds(name):
    out = {}
    for l in open(f"outputs/eval/{name}"):
        r = json.loads(l); out[r["id"]] = r
    return out

def text(d, xy, s, f, fill=INK, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)

def pill(d, xy, s, fg, bg):
    x, y = xy; w = d.textlength(s, font=F["tag"])
    d.rounded_rectangle([x, y, x+w+20, y+28], radius=14, fill=bg)
    d.text((x+10, y+5), s, font=F["tag"], fill=fg)
    return x+w+20


# labels.csv gives us, per uid: the full tile paths and the bbox, for the
# chipping figure and to pick clean examples.
def load_labels():
    rows = {}
    with open("data/labels.csv") as f:
        for r in csv.DictReader(f):
            rows[r["uid"]] = r
    return rows


# 7 ── before/after chip pairs, one clean example per grade ---------------
def fig_chip_pairs(labels):
    # pick a uid per grade that has both chips and reasonable size
    pick = {}
    for uid, r in labels.items():
        g = r["damage_grade"]
        if g in pick: continue
        if chip(uid, r["disaster"], "pre") and chip(uid, r["disaster"], "post") \
           and float(r["area_px"] or 0) > 300:
            pick[g] = (uid, r)
        if len(pick) == 4: break
    grades = ["no-damage","minor-damage","major-damage","destroyed"]
    C, pad, top, gap, lblw = 224, 26, 120, 20, 150
    W = lblw + pad + 2*C + gap + pad
    H = top + 4*(C+gap) + pad
    im = Image.new("RGB", (W, H), SURFACE); d = ImageDraw.Draw(im)
    text(d, (pad, 26), "Before / after chips, one example per grade", F["h"])
    text(d, (pad, 62), "The model's input. Each is a 448px crop around one building; shown here at 224.", F["cap"], MUTED)
    text(d, (lblw+pad, top-26), "PRE-EVENT", F["tag"], MUTED)
    text(d, (lblw+pad+C+gap, top-26), "POST-EVENT", F["tag"], MUTED)
    for i, g in enumerate(grades):
        y = top + i*(C+gap)
        uid, r = pick[g]
        d.rounded_rectangle([pad, y+C/2-30, lblw-10, y+C/2+30], radius=8, fill=SEV[g])
        text(d, (pad+14, y+C/2), g.replace("-","\n"), F["lab"],
             fill=(255,255,255) if g!="no-damage" else INK, anchor="lm")
        for j, ph in enumerate(["pre","post"]):
            c = chip(uid, r["disaster"], ph).resize((C, C))
            x = lblw+pad + j*(C+gap)
            im.paste(c, (x, y)); d.rectangle([x,y,x+C-1,y+C-1], outline=LINE, width=2)
    im.save(FIG/"07_chip_pairs.png"); print("wrote", FIG/"07_chip_pairs.png")
    return pick


# 8 ── inference cards: same chip, base vs tuned output -------------------
def fig_inference(labels):
    base = load_preds("base_post_gpu.jsonl")
    tuned = load_preds("run1_post.jsonl")
    # choose examples where base is wrong (its usual minor-damage) and tuned right,
    # one per non-minor grade for contrast.
    chosen = []
    want = ["destroyed","major-damage","no-damage"]
    for uid, r in labels.items():
        g = r["damage_grade"]
        if g not in want or g in [c[1] for c in chosen]: continue
        b, t = base.get(uid), tuned.get(uid)
        if not (b and t and chip(uid, r["disaster"])): continue
        bp = parse_response(b["generation"]); tp = parse_response(t["generation"])
        if bp.ok and tp.ok and bp.assessment.damage != g and tp.assessment.damage == g:
            chosen.append((uid, g, r, b, t))
        if len(chosen) == 3: break

    C, pad = 240, 30; colw = 430
    W = pad + C + 40 + 2*colw + pad
    rowh = C + 40
    H = 120 + len(chosen)*(rowh+24) + pad
    im = Image.new("RGB", (W, H), SURFACE); d = ImageDraw.Draw(im)
    text(d, (pad, 28), "Base vs tuned on the same building (3B, post-only)", F["h"])
    text(d, (pad, 66), "Base is a confident constant predictor. Tuning teaches it to actually grade. Green = matches ground truth.", F["cap"], MUTED)
    cx0 = pad+C+40
    text(d, (cx0, 100), "BASELINE (zero-shot)", F["tag"], MUTED)
    text(d, (cx0+colw, 100), "TUNED (LoRA)", F["tag"], MUTED)
    for i, (uid, g, r, b, t) in enumerate(chosen):
        y = 130 + i*(rowh+24)
        c = chip(uid, r["disaster"]).resize((C, C))
        im.paste(c, (pad, y)); d.rectangle([pad,y,pad+C-1,y+C-1], outline=LINE, width=2)
        d.rounded_rectangle([pad, y+C+6, pad+C, y+C+34], radius=8, fill=SEV[g])
        text(d, (pad+C/2, y+C+20), f"truth: {g}", F["tag"],
             fill=(255,255,255) if g!="no-damage" else INK, anchor="mm")
        for j, (pred_r, ) in enumerate([(b,), (t,)]):
            pr = parse_response(pred_r["generation"]); a = pr.assessment
            x = cx0 + j*colw
            right = (a.damage == g)
            d.rounded_rectangle([x, y, x+colw-30, y+rowh], radius=12,
                                outline=OK if right else LINE, width=3 if right else 1,
                                fill=(247,251,249) if right else SURFACE)
            xx = pill(d, (x+16, y+14), a.damage, (255,255,255) if a.damage!="no-damage" else INK, SEV[a.damage])
            if right:
                cyc = y+28
                d.line([(xx+16,cyc),(xx+23,cyc+7),(xx+35,cyc-8)], fill=OK, width=4)
                text(d, (xx+44, y+18), "correct", F["sm"], OK)
            else:
                text(d, (xx+16, y+18), "wrong", F["sm"], BAD)
            # wrap evidence
            ev = a.evidence; words = ev.split(); lines=[]; cur=""
            for w in words:
                if d.textlength(cur+" "+w, font=F["sm"]) < colw-70: cur=(cur+" "+w).strip()
                else: lines.append(cur); cur=w
            lines.append(cur)
            for li, ln in enumerate(lines[:5]):
                text(d, (x+18, y+56+li*22), ln, F["sm"], INK)
            text(d, (x+18, y+rowh-30), f"priority: {a.priority}", F["sm"], MUTED)
    im.save(FIG/"08_inference_cards.png"); print("wrote", FIG/"08_inference_cards.png")


# 9 ── chipping: full tile -> chip, with the building located ------------
def fig_chipping(labels, pick):
    uid, r = pick["destroyed"]
    post_tile = Image.open(r["post_image"]).convert("RGB")
    pre_tile = Image.open(r["pre_image"]).convert("RGB") if r["pre_image"] else post_tile
    bx0,by0,bx1,by1 = (float(r[k]) for k in ["bbox_x0","bbox_y0","bbox_x1","bbox_y1"])
    T = 380
    def prep(tile, box=False):
        t = tile.copy(); dr = ImageDraw.Draw(t)
        if box:
            dr.rectangle([bx0,by0,bx1,by1], outline=(255,80,80), width=6)
            # context window (3x)
            cx,cy=float(r["centroid_x"]),float(r["centroid_y"])
            side=max(bx1-bx0,by1-by0)*3
            dr.rectangle([cx-side/2,cy-side/2,cx+side/2,cy+side/2], outline=(90,160,240), width=5)
        return t.resize((T,T))
    pre_s, post_s = prep(pre_tile), prep(post_tile, box=True)
    chip_img = chip(uid, r["disaster"]).resize((T,T))
    pad=30; gap=40; W=pad+3*T+2*gap+pad; H=150+T+70
    im=Image.new("RGB",(W,H),SURFACE); d=ImageDraw.Draw(im)
    text(d,(pad,28),"From full tile to building chip",F["h"])
    text(d,(pad,66),f"{r['disaster']}  ·  1024px tiles  ·  red = building polygon, blue = 3× context crop that becomes the 448px chip.",F["cap"],MUTED)
    xs=[pad, pad+T+gap, pad+2*(T+gap)]
    labs=["pre-event tile","post-event tile (target located)","extracted post chip"]
    for x,img,lab in zip(xs,[pre_s,post_s,chip_img],labs):
        im.paste(img,(x,120)); d.rectangle([x,120,x+T-1,120+T-1],outline=LINE,width=2)
        text(d,(x,120+T+14),lab,F["cap"],MUTED)
    # arrow
    for xa in [pad+T+gap//2-10, pad+2*T+gap+gap//2-10]:
        d.line([xa-8,120+T//2,xa+12,120+T//2],fill=MUTED,width=3)
        d.polygon([(xa+12,120+T//2-7),(xa+22,120+T//2),(xa+12,120+T//2+7)],fill=MUTED)
    im.save(FIG/"09_chipping.png"); print("wrote",FIG/"09_chipping.png")


if __name__ == "__main__":
    labels = load_labels()
    pick = fig_chip_pairs(labels)
    fig_inference(labels)
    fig_chipping(labels, pick)
    print("\nimage composites ->", FIG)
