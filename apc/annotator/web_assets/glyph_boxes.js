function applyGlyphBox(annotation, type, index, box) {
  const match = /^(hero|board)_(rank|suit)$/.exec(type);
  if (!match || !Number.isInteger(index) || index < 0) throw new Error("Invalid glyph target");
  const card = annotation.objects?.[match[1] + "_cards"]?.[index];
  if (!card?.box) throw new Error("Draw the parent card first, using the same array index.");
  const parent = card.box;
  if (![box.x, box.y, box.width, box.height].every(Number.isFinite) ||
      box.width <= 0 || box.height <= 0 ||
      box.x < parent.x || box.y < parent.y ||
      box.x + box.width > parent.x + parent.width + 1e-9 ||
      box.y + box.height > parent.y + parent.height + 1e-9) {
    throw new Error("Rank and suit regions must remain inside the parent card.");
  }
  card[match[2] + "_box"] = {...box};
  return annotation;
}
if (typeof module !== "undefined") module.exports = {applyGlyphBox};
