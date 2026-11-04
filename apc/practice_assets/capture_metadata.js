function buildVisibleFrameMetadata(state, viewport, regions) {
  if (!state || state.units !== "BB") throw new Error("A BB table state is required");
  if (!(viewport.width > 0 && viewport.height > 0)) throw new Error("Invalid viewport");
  const publicState = {};
  for (const key of ["hand_id","revision","state_fingerprint","street","hero_cards","board","stacks_bb",
                     "pot_bb","to_call_bb","hero_position","button_position","action_history","terminal"]) {
    publicState[key] = structuredClone(state[key]);
  }
  return {schema_version:"1.0.0",source:"apc_full_hand_browser",units:"BB",draft_only:true,
    screenshot_required:true,image_hash_required:true,independent_review_required:true,
    viewport:structuredClone(viewport),public_state:publicState,regions:structuredClone(regions),
    excluded:["opponent_private_cards","automatic_readiness_counts"]};
}
if(typeof module !== "undefined") module.exports={buildVisibleFrameMetadata};
