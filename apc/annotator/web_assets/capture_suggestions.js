/* Pure draft transformation: never saves, verifies, or invents table state. */
function applyCaptureSuggestion(annotation, suggestion) {
  if (suggestion?.kind !== "apc_capture_metadata_suggestion" || suggestion.review_required !== true || suggestion.auto_applied !== false) {
    throw new Error("Expected a review-required capture suggestion.");
  }
  const result = structuredClone(annotation);
  result.objects = result.objects || {};
  let count = 0;
  for (const [collection, labelKey] of [["hero_cards","hero_cards"],["board_cards","board"]]) {
    const rows = suggestion.suggested_objects?.[collection];
    const labels = suggestion.draft_card_labels?.[labelKey];
    if (!Array.isArray(rows) || !Array.isArray(labels) || rows.length !== labels.length) {
      throw new Error("Capture card boxes and identities must correspond.");
    }
    result.objects[collection] = rows.map((row,index) => {
      const box = row?.box, identity = labels[index];
      if (!box || ![box.x,box.y,box.width,box.height].every(Number.isFinite)
          || box.x < 0 || box.y < 0 || box.width <= 0 || box.height <= 0
          || box.x+box.width > 1 || box.y+box.height > 1 || !/^[2-9TJQKA][shdc]$/.test(identity)) {
        throw new Error("Invalid capture card geometry or identity.");
      }
      count++;
      return {box:structuredClone(box),rank:identity[0],suit:identity[1],visibility:"uncertain"};
    });
  }
  if (!count) throw new Error("Capture suggestion has no visible cards.");
  result.provenance = result.provenance || {annotator:"human-review",annotation_version:1,created_at:new Date().toISOString()};
  result.provenance.verified = false;
  result.provenance.reviewer = null;
  const marker = `capture_suggestion_sha256=${suggestion.suggestion_sha256}; human review required`;
  const notes = result.provenance.notes || "";
  if (!notes.includes(marker)) result.provenance.notes = `${notes}${notes ? "\n" : ""}${marker}`;
  return result;
}
if (typeof module !== "undefined") module.exports = {applyCaptureSuggestion};
