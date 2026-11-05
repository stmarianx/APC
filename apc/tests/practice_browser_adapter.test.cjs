const assert = require('node:assert/strict');
(async () => {
  const {capturePracticeFrame} = await import('../capture/practice_browser_adapter.mjs');
  const events = [];
  const snapshot = {viewport:[800,900],scroll:[0,0],cards:[['hero','As',100,200,62,78]]};
  const tab = {
    playwright:{evaluate:async fn => {assert.equal(typeof fn,'function');events.push('observe');return snapshot;}},
    screenshot:async opts => {assert.deepEqual(opts,{fullPage:false});events.push('screenshot');snapshot.scroll[1]=80;return new Uint8Array([1,2]);}
  };
  const result = await capturePracticeFrame(tab);
  assert.deepEqual(events,['observe','screenshot','observe']);
  assert.deepEqual(result.before.scroll,[0,0]);
  assert.deepEqual(result.after.scroll,[0,80]);
  assert.deepEqual(Array.from(result.png),[1,2]);
  console.log('Practice adapter orchestration tests passed');
})().catch(error => {console.error(error);process.exitCode=1;});
