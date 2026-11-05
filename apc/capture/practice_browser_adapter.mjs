/* Read-only adapter for APC's own practice DOM, not a platform perception model.
 * Call only through an authorized browser session. No navigation/game actions.
 * Python capture_stability remains the authoritative evidence validator.
 */
async function observePracticeFrame(tab) {
  return tab.playwright.evaluate(() => {
    const table = document.querySelector('#table[aria-label="Virtual poker table"]');
    if (!table) throw new Error('APC practice table not visibly present');
    const suits = {'♠':'s','♥':'h','♦':'d','♣':'c'};
    const cards = Array.from(document.querySelectorAll('#board .card,#heroCards .card')).map(e => {
      const style = getComputedStyle(e), r = e.getBoundingClientRect();
      if (style.display === 'none' || style.visibility !== 'visible' || Number(style.opacity) === 0
          || r.width <= 0 || r.height <= 0 || r.x < 0 || r.y < 0
          || r.right > innerWidth || r.bottom > innerHeight) {
        throw new Error('Practice card not fully visible');
      }
      const text = e.textContent.trim().replace(/\s/g, '');
      const identity = text.slice(0, -1).replace('10', 'T') + suits[text.slice(-1)];
      if (!/^[2-9TJQKA][shdc]$/.test(identity)) throw new Error('Unrecognized visible card');
      return [e.parentElement.id === 'board' ? 'board' : 'hero', identity, r.x, r.y, r.width, r.height];
    });
    return {viewport:[innerWidth,innerHeight],scroll:[scrollX,scrollY],cards};
  });
}

async function capturePracticeFrame(tab) {
  const before = structuredClone(await observePracticeFrame(tab));
  const png = await tab.screenshot({fullPage:false});
  const after = structuredClone(await observePracticeFrame(tab));
  // Return both observations even when different: retain, do not retry/select.
  return {before, png, after};
}

async function observePracticeAlignmentFrame(tab) {
  const frame = await observePracticeFrame(tab);
  frame.alignment_landmarks = await tab.playwright.evaluate(() => {
    const element = document.querySelector('#table[aria-label="Virtual poker table"]');
    const r = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    if (style.visibility !== 'visible' || r.x < 0 || r.y < 0
        || r.right > innerWidth || r.bottom > innerHeight) {
      throw new Error('Full practice table must be visible for alignment landmarks');
    }
    // Outer border extrema, not rounded bounding-box corners. Pixel locations
    // must be independently measured; this records only visible DOM geometry.
    return [
      {description:'table outer border top midpoint',dom_xy:[r.x+r.width/2,r.y]},
      {description:'table outer border right midpoint',dom_xy:[r.right,r.y+r.height/2]},
      {description:'table outer border bottom midpoint',dom_xy:[r.x+r.width/2,r.bottom]},
      {description:'table outer border left midpoint',dom_xy:[r.x,r.y+r.height/2]},
    ];
  });
  return frame;
}

async function capturePracticeAlignmentFrame(tab) {
  const before = structuredClone(await observePracticeAlignmentFrame(tab));
  const png = await tab.screenshot({fullPage:false});
  const after = structuredClone(await observePracticeAlignmentFrame(tab));
  return {before,png,after};
}

async function observePracticePublicFrame(tab) {
  const frame = await observePracticeAlignmentFrame(tab);
  frame.public_regions = await tab.playwright.evaluate(() => {
    const region = (element) => {
      const r = element.getBoundingClientRect(), style = getComputedStyle(element);
      if (style.display === 'none' || style.visibility !== 'visible' || Number(style.opacity) === 0
          || r.width <= 0 || r.height <= 0 || r.x < 0 || r.y < 0
          || r.right > innerWidth || r.bottom > innerHeight) {
        throw new Error('Public label region not fully visible');
      }
      return {text:element.innerText.trim(),box_xywh:[r.x,r.y,r.width,r.height]};
    };
    const bb = (text) => {
      if (!/^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$/.test(text)) throw new Error('Expected visible nonnegative BB number');
      return text;
    };
    const seats = ['hero','villain'].map((id,index) => {
      const label = region(document.getElementById(id));
      const parts = label.text.split('·').map(s=>s.trim());
      const stack = parts.find(part=>/ BB$/.test(part));
      if (!stack) throw new Error('Visible stack label missing');
      return {...region(document.getElementById(id).parentElement),seat_no:index+1,
              player_name:parts[0],stack_bb:bb(stack.slice(0,-3)),is_hero:id==='hero',
              has_dealer_button:parts.includes('Dealer')};
    });
    if (seats.filter(s=>s.has_dealer_button).length !== 1) throw new Error('Visible dealer label ambiguous');
    const pot = region(document.getElementById('pot'));
    const buttons = Array.from(document.querySelectorAll('#buttons button')).map(element=>{
      const row = region(element), action = row.text.split(' ')[0];
      if (!['fold','check','call','bet','raise','all_in'].includes(action)) throw new Error('Unsupported visible action button');
      return {...row,action,enabled:!element.disabled};
    });
    const street = region(document.getElementById('street')).text;
    if (!['preflop','flop','turn','river'].includes(street)) throw new Error('Unsupported visible street');
    const price = region(document.getElementById('price')).text;
    const match = /^To call ((?:0|[1-9][0-9]*)(?:\.[0-9]+)?) BB$/.exec(price);
    if (!match) throw new Error('Expected active-hand visible call price');
    // No window.state, network endpoints, exported metadata, or opponent cards.
    return {source:'owned_practice_visible_dom_v1',units:'BB',seats,pot:{...pot,amount_bb:bb(pot.text)},
            action_buttons:buttons,street,to_call_bb:bb(match[1])};
  });
  return frame;
}

async function capturePracticePublicFrame(tab) {
  const before = structuredClone(await observePracticePublicFrame(tab));
  const png = await tab.screenshot({fullPage:false});
  const after = structuredClone(await observePracticePublicFrame(tab));
  return {before,png,after};
}

async function observeNegativeFrame(tab) {
  return tab.playwright.evaluate(() => {
    const fixture = document.querySelector('main[data-apc-audit-negative]');
    if (!fixture || !['text','lobby','controls'].includes(fixture.getAttribute('data-apc-audit-negative'))
        || !fixture.textContent.trim() || document.querySelector('.card')) {
      throw new Error('Expected explicit card-free APC audit fixture');
    }
    return {viewport:[innerWidth,innerHeight],scroll:[scrollX,scrollY],cards:[],
      fixture_scene:fixture.getAttribute('data-apc-audit-negative')};
  });
}

async function captureNegativeFrame(tab) {
  const before = structuredClone(await observeNegativeFrame(tab));
  const png = await tab.screenshot({fullPage:false});
  const after = structuredClone(await observeNegativeFrame(tab));
  return {before,png,after};
}

export {observePracticeFrame, capturePracticeFrame, observeNegativeFrame, captureNegativeFrame};
export {observePracticeAlignmentFrame, capturePracticeAlignmentFrame};
export {observePracticePublicFrame, capturePracticePublicFrame};
