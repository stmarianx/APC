const assert = require("node:assert/strict");
const {applyGlyphBox} = require("../annotator/web_assets/glyph_boxes.js");
for (const role of ["hero", "board"]) {
  const annotation = {objects: {[role+"_cards"]: [{box: {x:.1,y:.1,width:.2,height:.3},rank:"A",suit:"s"}]}};
  for (const glyph of ["rank", "suit"]) {
    const box = {x:.12,y:.12,width:.05,height:.05};
    applyGlyphBox(annotation, role+"_"+glyph, 0, box);
    assert.deepEqual(annotation.objects[role+"_cards"][0][glyph+"_box"], box);
    box.x = 0;
    assert.equal(annotation.objects[role+"_cards"][0][glyph+"_box"].x, .12);
  }
  assert.equal(annotation.objects[role+"_cards"][0].rank, "A");
  for (const box of [{x:0,y:0,width:1,height:1},{x:.1,y:.1,width:0,height:.1},{x:NaN,y:.1,width:.1,height:.1}]) {
    assert.throws(() => applyGlyphBox(annotation, role+"_rank", 0, box));
  }
  assert.throws(() => applyGlyphBox(annotation, role+"_rank", 2, {x:.1,y:.1,width:.1,height:.1}));
}
console.log("Glyph labeling tests passed");
