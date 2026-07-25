#!/usr/bin/env python3
"""Render the context-conditioning demo figure from demo_conditioning.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbd_vlm.schema import parse_response

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)
INK,MUTED,SURF,LINE,CARD = (34,32,29),(107,102,94),(250,248,243),(224,220,211),(255,255,255)
SEV = {"no-damage":(120,150,110),"minor-damage":(224,139,63),
       "major-damage":(200,90,60),"destroyed":(127,43,43)}
PRIO = {"none":(120,150,110),"moderate":(210,150,60),"high":(200,90,60),"critical":(150,40,40)}
HL = (233,225,150)  # highlight for the conditioned phrase

def font(sz,b=False):
    for p in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if b else
               "/System/Library/Fonts/Supplemental/Arial.ttf"]):
        try: return ImageFont.truetype(p,sz)
        except OSError: pass
    return ImageFont.load_default()
F={k:font(s,b) for k,(s,b) in {"h":(31,True),"sub":(17,False),"sec":(20,True),
    "ctx":(17,True),"pill":(15,True),"ev":(15,False),"lab":(13,True),"big":(18,True)}.items()}

d0=json.load(open("outputs/eval/demo_conditioning.json"))
def parsed(res): return [(r,parse_response(r["generation"]).assessment) for r in res]
tuned=parsed(d0["tuned"]); base=parsed(d0["base"])

def pill(dr,x,y,s,fg,bg,f=None):
    f=f or F["pill"]; w=dr.textlength(s,font=f)
    dr.rounded_rectangle([x,y,x+w+22,y+26],radius=13,fill=bg)
    dr.text((x+11,y+4),s,font=f,fill=fg); return x+w+22

def wrap(dr,s,f,maxw):
    words=s.split(); lines=[]; cur=""
    for w in words:
        if dr.textlength(cur+" "+w,font=f)<maxw: cur=(cur+" "+w).strip()
        else: lines.append(cur); cur=w
    lines.append(cur); return lines

W=1120
im=Image.new("RGB",(W,1650),SURF); d=ImageDraw.Draw(im)
d.text((40,32),"Context conditioning: same pixels, different reading",font=F["h"],fill=INK)
sub=("The image never changes; only the stated context does. The base model ignores it. The tuned model's assessment moves with the words — "
     "the thing a classifier structurally cannot do — and in moving, reveals that the conditioning was trained in, not discovered.")
for i,ln in enumerate(wrap(d,sub,F["sub"],W-80)[:2]):
    d.text((40,76+i*22),ln,font=F["sub"],fill=MUTED)

# fixed image
IMG=300; ix,iy=40,150
chip=Image.open(d0["chip"]).convert("RGB").resize((IMG,IMG))
im.paste(chip,(ix,iy)); d.rectangle([ix,iy,ix+IMG-1,iy+IMG-1],outline=INK,width=3)
d.rounded_rectangle([ix,iy+IMG+10,ix+IMG,iy+IMG+40],radius=8,fill=INK)
d.text((ix+IMG/2,iy+IMG+25),"FIXED IMAGE — never changes",font=F["lab"],fill=(255,255,255),anchor="mm")
d.text((ix,iy+IMG+52),f"{d0['disaster']}, truth = major-damage",font=F["ev"],fill=MUTED)

# base contrast box
bx=ix; by=iy+IMG+90
d.rounded_rectangle([bx,by,bx+IMG,by+150],radius=12,fill=CARD,outline=LINE)
d.text((bx+16,by+14),"BASE MODEL",font=F["lab"],fill=MUTED)
d.text((bx+16,by+38),"all 7 variants:",font=F["ev"],fill=INK)
pill(d,bx+16,by+64,"minor-damage",(255,255,255),SEV["minor-damage"])
pill(d,bx+16,by+98,"moderate",(255,255,255),PRIO["moderate"])
d.text((bx+16,by+130),"context ignored.",font=F["ev"],fill=MUTED)

# right column: variants
cx=380; cw=W-cx-40
def section(y,title,note):
    d.text((cx,y),title,font=F["sec"],fill=INK)
    lines=wrap(d,note,F["ev"],cw)
    for i,ln in enumerate(lines[:2]):
        d.text((cx,y+30+i*18),ln,font=F["ev"],fill=MUTED)
    return y+50+len(lines[:2])*18

def row(y,ctx_label,a,changed_phrase):
    h=104
    d.rounded_rectangle([cx,y,cx+cw,y+h],radius=12,fill=CARD,outline=LINE)
    # context
    d.rounded_rectangle([cx+14,y+16,cx+190,y+46],radius=8,fill=INK)
    d.text((cx+22,y+31),ctx_label,font=F["ctx"],fill=(255,255,255),anchor="lm")
    # damage + priority pills
    xx=pill(d,cx+210,y+18,a.damage,(255,255,255),SEV[a.damage])
    pill(d,xx+10,y+18,a.priority,(255,255,255),PRIO[a.priority])
    # evidence with the conditioned phrase highlighted
    ev=a.evidence
    lines=wrap(d,ev,F["ev"],cw-40)
    for li,ln in enumerate(lines[:3]):
        ty=y+52+li*20
        # highlight the changed phrase if present in this line
        if changed_phrase and changed_phrase.lower() in ln.lower():
            s=ln.lower().index(changed_phrase.lower())
            pre=ln[:s]; mid=ln[s:s+len(changed_phrase)]
            px=cx+18+d.textlength(pre,font=F["ev"])
            d.rectangle([px-2,ty-1,px+d.textlength(mid,font=F["ev"])+2,ty+17],fill=HL)
        d.text((cx+18,ty),ln,font=F["ev"],fill=INK)
    return y+h+14

y=section(150,"1 — Change the event type","Days-since fixed at 5. The evidence clause (highlighted) tracks the event, and for wildfire the grade itself drops to no-damage.")
phrases={"hurricane":"prevailing wind direction",
         "flood":"standing water",
         "wildfire":"charring on adjacent parcels",
         "earthquake":"comparable collapse patterns"}
for (r,a) in tuned[:4]:
    ev=r["event_type"]; y=row(y,ev,a,phrases.get(ev,""))

y+=14
y=section(y,"2 — Change the days-since","Event fixed at hurricane. Priority tracks days by design — but the grade shifts too, which it should not on identical pixels. A learned spurious coupling.")
for (r,a) in tuned[4:]:
    y=row(y,f"{r['days']} days",a,None)

# honest footer
y+=8
foot=("Honest read: the base model already parrots event-appropriate words, so evidence text alone is not proof. The real tell is that the tuned model's grade and "
      "priority move with the context while the base model's never do — and the days→grade shift exposes conditioning learned from templated targets, not reasoning.")
flines=wrap(d,foot,F["ev"],cw-32)
fh=18+len(flines)*20
d.rounded_rectangle([cx,y,cx+cw,y+fh],radius=10,fill=(245,240,228),outline=LINE)
for i,ln in enumerate(flines):
    d.text((cx+16,y+13+i*20),ln,font=F["ev"],fill=INK)

im=im.crop((0,0,W,y+fh+30))
im.save(FIG/"10_context_conditioning.png")
print("wrote",FIG/"10_context_conditioning.png")
