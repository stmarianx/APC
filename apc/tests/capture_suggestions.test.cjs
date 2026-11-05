const {test} = require('node:test');
const assert = require('node:assert/strict');
const {applyCaptureSuggestion} = require('../annotator/web_assets/capture_suggestions.js');
const suggestion = () => ({kind:'apc_capture_metadata_suggestion',review_required:true,auto_applied:false,
  suggestion_sha256:'fixture',suggested_objects:{hero_cards:[{box:{x:.2,y:.3,width:.1,height:.2}}],board_cards:[]},
  draft_card_labels:{hero_cards:['As'],board:[]}});

test('capture cards apply to a cloned draft without inventing full hand state or approving',()=>{
  const source={objects:{pot:{amount_bb:'4'}},state:{hand_id:'kept'},provenance:{verified:true,reviewer:'old'}};
  const original=structuredClone(source), s=suggestion(), originalSuggestion=structuredClone(s);
  const result=applyCaptureSuggestion(source,s);
  assert.deepEqual(source,original); assert.deepEqual(s,originalSuggestion);
  assert.deepEqual(result.state,{hand_id:'kept'});
  assert.deepEqual(result.objects.pot,{amount_bb:'4'});
  assert.equal(result.objects.hero_cards[0].rank,'A');
  assert.equal(result.objects.hero_cards[0].suit,'s');
  assert.equal(result.objects.hero_cards[0].visibility,'uncertain');
  assert.equal(result.provenance.verified,false); assert.equal(result.provenance.reviewer,null);
  assert.match(result.provenance.notes,/human review required/);
  assert.deepEqual(result,applyCaptureSuggestion(result,s));
});
test('malformed capture suggestion never partially modifies input',()=>{
  for(const corrupt of [s=>s.auto_applied=true,s=>s.draft_card_labels.hero_cards=[],
    s=>s.draft_card_labels.hero_cards=['XX'],s=>s.suggested_objects.hero_cards[0].box.x=.99,
    s=>s.suggested_objects.hero_cards[0].box.width=NaN]) {
    const s=suggestion(); corrupt(s); const original={objects:{},state:{}};
    assert.throws(()=>applyCaptureSuggestion(original,s));
    assert.deepEqual(original,{objects:{},state:{}});
  }
});
test('visible public fields and legacy price/pot labels remain review-only',()=>{
  const s=suggestion();
  s.suggested_visible_state={street:'preflop',pot_bb:'1.5',to_call_bb:'0.5',hero_seat:1,dealer_seat:1,
    legal_actions:['fold','call'],hand_id:'must-not-copy'};
  s.suggested_objects.seats=[{seat_no:1,stack_bb:'99.5',player_name:'Hero',box:{x:.2,y:.3,width:.1,height:.2}}];
  s.suggested_objects.pot={amount_bb:'1.5',box:{x:.4,y:.3,width:.1,height:.05}};
  s.suggested_objects.action_buttons=[{action:'call',enabled:true,box:{x:.3,y:.8,width:.1,height:.05}}];
  const result=applyCaptureSuggestion({state:{hand_id:'existing'},objects:{}},s);
  assert.equal(result.state.hand_id,'existing');
  assert.equal(result.state.to_call_bb,'0.5');
  assert.equal(result.objects.seats[0].stack_bb,'99.5');
  assert.equal(result.objects.pot.amount_bb,'1.5');
  assert.equal(result.objects.action_buttons[0].action,'call');
  assert.equal(result.provenance.verified,false);
  assert.equal(result.objects.seats[0].visibility,'uncertain');
});
