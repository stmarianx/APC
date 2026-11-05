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

export {observePracticeFrame, capturePracticeFrame};
