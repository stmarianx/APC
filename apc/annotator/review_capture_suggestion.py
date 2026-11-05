"""Render all available draft capture-object boxes, never approve annotations."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from apc.annotator.project import AnnotationProject


def render_suggestion(project, sample_id, output):
    record=project.record(sample_id)
    suggestion=project.load_suggestion(sample_id)
    if not suggestion or suggestion.get("kind") != "apc_capture_metadata_suggestion":
        raise ValueError("Expected an unverified capture suggestion")
    source=(project.root/record.frame_path).resolve()
    if not source.is_relative_to(project.root) or hashlib.sha256(source.read_bytes()).hexdigest() != record.sha256:
        raise ValueError("Suggestion source image changed")
    with Image.open(source) as opened:
        image=opened.convert("RGB")
    objects=suggestion["suggested_objects"]
    rows=[]
    for key,color in (("hero_cards","#ffdf00"),("board_cards","#ffdf00"),
                      ("seats","#00ffff"),("action_buttons","#ff50ff")):
        for index,row in enumerate(objects.get(key,[])):
            label=key+":"+str(index+1)
            if key == "seats": label=f"seat {row['seat_no']} {row['stack_bb']} BB"
            if key == "action_buttons": label=row["action"]
            rows.append((label,color,row["box"]))
    if "pot" in objects:
        rows.append(("pot "+objects["pot"].get("amount_bb","")+" BB","#ffa000",objects["pot"]["box"]))
    draw=ImageDraw.Draw(image)
    rendered=[]
    for label,color,box in rows:
        coords=[round(box["x"]*image.width),round(box["y"]*image.height),
                round((box["x"]+box["width"])*image.width),round((box["y"]+box["height"])*image.height)]
        draw.rectangle(coords,outline=color,width=2)
        draw.text((coords[0],max(0,coords[1]-13)),label,fill=color,stroke_width=1,stroke_fill="black")
        rendered.append({"label":label,"pixel_xyxy":coords})
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    image.save(output/"overlay.png")
    report={"sample_id":sample_id,"source_image_sha256":record.sha256,
            "suggestion_sha256":suggestion["suggestion_sha256"],"objects":rendered,
            "skipped_unmeasured_regions":suggestion.get("skipped_unmeasured_public_regions",[]),
            "review_status":"pending","independent_review":False,"readiness_credit":False,"model_used":False}
    (output/"review.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--sample-id")
    args=parser.parse_args()
    project=AnnotationProject(args.project)
    if args.sample_id is None and len(project.records) != 1:
        parser.error("--sample-id required unless project contains exactly one frame")
    render_suggestion(project,args.sample_id or project.records[0].sample_id,args.output)
