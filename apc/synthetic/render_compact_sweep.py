"""Deterministic geometry-development sweep; no captured images or hidden labels."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from apc.synthetic.render_table import _draw_suit, SUIT_COLORS


def render_sweep(output,font_path,seed=20261106):
    output, font_path = Path(output),Path(font_path)
    font = ImageFont.truetype(str(font_path),18)
    font_hash = hashlib.sha256(font_path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False)
    (output / "frames").mkdir()
    rng = random.Random(seed)
    frames = []
    conditions = list(itertools.product((560,800),(-12,-4,4,12),(2,6,12),(220,540)))
    for index,(width,shift,gap,y) in enumerate(conditions):
        image = Image.new("RGB",(width,900),"#101b19")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((12,130,width-12,680),radius=80,fill="#205440",outline="#494137",width=8)
        x = width//2-44-gap//2+shift
        draw.rounded_rectangle((x-20,y-12,x+108+gap,y+94),radius=10,fill="#14291f",outline="#658773")
        deck = rng.sample([r+s for r in "23456789TJQKA" for s in "cdhs"],2)
        cards = []
        for offset,identity in enumerate(deck):
            cx = x+offset*(44+gap)
            draw.rounded_rectangle((cx,y,cx+43,y+63),radius=5,fill="#f5f3e9")
            color = SUIT_COLORS[identity[1]]
            draw.text((cx+6,y+6),identity[0],fill=color,font=font,anchor="lt")
            _draw_suit(draw,(cx+8,y+33,cx+22,y+49),identity[1],color)
            cards.append(["hero",identity,cx,y])
        draw.text((20,740),"Fold     Check     Raise 2.5 BB",font=font,fill="#c9ed9a")
        name = f"frames/positive-{index:03d}.png"
        image.save(output / name)
        frames.append({"image":name,"sha256":hashlib.sha256((output/name).read_bytes()).hexdigest(),
            "hand_group":f"synthetic-{index:03d}","street":"preflop","viewport":[width,900],"cards":cards,
            "sweep_condition":{"x_shift":shift,"gap":gap,"y":y}})
    for index,(control_width,background,placement) in enumerate(itertools.product((20,32,44),("#101b19","#205440"),(0,1))):
        image = Image.new("RGB",(800,900),background)
        draw = ImageDraw.Draw(image)
        x,y = (32,620) if placement == 0 else (480,270)
        draw.rounded_rectangle((x-8,y-8,x+180,y+120),radius=8,fill="#deded2")
        for offset,label in enumerate(("Stack 100 BB","Player","Call 2.5 BB")):
            draw.text((x,y+offset*30),label,font=font,fill="#172a22",anchor="lt")
        # Explicit card-like narrow control with no playing-card glyph pair.
        draw.rounded_rectangle((x+210,y,x+210+control_width,y+70),radius=4,fill="#f5f3e9")
        draw.text((x+212,y+8),"OK",font=font,fill="#172a22",anchor="lt")
        name = f"frames/negative-{index:03d}.png"
        image.save(output/name)
        frames.append({"image":name,"sha256":hashlib.sha256((output/name).read_bytes()).hexdigest(),
            "hand_group":f"negative-{index:03d}","street":"empty","viewport":[800,900],"cards":[],
            "sweep_condition":{"control_width":control_width,"background":background,"placement":placement}})
    data = {"schema_version":"apc_browser_card_development_v1","capture_session_id":f"synthetic-compact-sweep-{seed}",
        "source":"synthetic_compact_geometry_sweep_v1","virtual_chips":True,"draft_only":True,
        "readiness_credit":False,"use_policy":"synthetic_development_only","viewport":[800,900],
        "card_size":[44,64],"rank_region_offset_xywh":[6,6,32,24],"suit_region_offset_xywh":[6,30,32,24],
        "screenshot_mapping":"normalized_viewport","seed":seed,"font_sha256":font_hash,
        "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"frames":frames}
    (output/"collection.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--font",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20261106)
    args = parser.parse_args()
    result = render_sweep(args.output,args.font,args.seed)
    print(json.dumps({"frames":len(result["frames"]),"targets":sum(len(r["cards"]) for r in result["frames"])}))
