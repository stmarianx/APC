/* Persist authorized adapter output; no browser access or game actions here. */
import {createHash} from 'node:crypto';
import {mkdir, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {isDeepStrictEqual} from 'node:util';

function buildAcquisition(acquired, {usePolicy = 'draft_review_no_training'} = {}) {
  if (!['draft_review_no_training','sealed_audit_no_training'].includes(usePolicy)) {
    throw new Error('Explicit supported capture use policy required');
  }
  const png = Buffer.from(acquired.png);
  const signature = Buffer.from([137,80,78,71,13,10,26,10]);
  let imageSize, format;
  if (png.length >= 33 && png.subarray(0,8).equals(signature)
      && png.readUInt32BE(8) === 13 && png.toString('ascii',12,16) === 'IHDR') {
    imageSize = [png.readUInt32BE(16),png.readUInt32BE(20)]; format = 'PNG';
  } else if (png.length >= 4 && png[0] === 255 && png[1] === 216) {
    // Read JPEG SOF dimensions without transcoding or modifying original bytes.
    let offset = 2;
    while (offset < png.length) {
      if (png[offset++] !== 255) break;
      while (offset < png.length && png[offset] === 255) offset++;
      const marker = png[offset++];
      if (marker === 218 || marker === 217) break;
      if (marker === 1 || (marker >= 208 && marker <= 215)) continue;
      if (offset+2 > png.length) break;
      const length = png.readUInt16BE(offset);
      if (length < 2 || offset+length > png.length) break;
      if ([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(marker) && length >= 8) {
        imageSize = [png.readUInt16BE(offset+5),png.readUInt16BE(offset+3)]; format='JPEG'; break;
      }
      offset += length;
    }
  }
  if (!imageSize) throw new Error('PNG or JPEG screenshot header required');
  if (imageSize.some(v => v <= 0)) throw new Error('Invalid PNG dimensions');
  const sha256 = createHash('sha256').update(png).digest('hex');
  const before = structuredClone(acquired.before), after = structuredClone(acquired.after);
  const numbers = (a,n) => Array.isArray(a) && a.length === n && a.every(v => typeof v === 'number' && Number.isFinite(v));
  const validSnapshot = s => s && numbers(s.viewport,2) && s.viewport.every(v => v > 0)
    && numbers(s.scroll,2) && Array.isArray(s.cards) && s.cards.every(c => Array.isArray(c)
      && c.length === 6 && ['hero','board'].includes(c[0]) && /^[2-9TJQKA][shdc]$/.test(c[1])
      && numbers(c.slice(2),4) && c[4] > 0 && c[5] > 0);
  let rejection = null;
  if (!validSnapshot(before) || !validSnapshot(after)) rejection = 'Invalid capture snapshot';
  else if (!isDeepStrictEqual(before,after)) rejection = 'Capture moved during screenshot';
  else {
    const extentHeight = imageSize[1]*before.viewport[0]/imageSize[0];
    if (before.cards.some(([, ,x,y,w,h]) => x < 0 || y < 0 || x+w > before.viewport[0] || y+h > extentHeight)) {
      rejection = 'Card outside saved screenshot';
    } else if (new Set(before.cards.map(c => c[1])).size !== before.cards.length) {
      rejection = 'Duplicate visible identity';
    }
  }
  const cards = Array.isArray(before?.cards) ? before.cards.filter(Array.isArray).map(c => c.slice(0,4)) : [];
  const metadata = {schema_version:'apc_browser_frame_acquisition_v1',image:format === 'PNG' ? 'frame.png' : 'frame.jpg',image_format:format,
    sha256,image_size:imageSize,cards,use_policy:usePolicy,
    capture_stability:{schema_version:'apc_capture_stability_v1',image_sha256:sha256,image_size:imageSize,before,after},
    eligible_for_draft_scoring:rejection === null,rejection_reason:rejection,
    pixel_geometry_reviewed:false,readiness_credit:false};
  // JSON cannot safely retain NaN/Infinity/undefined; reject before persistence.
  const serialized = JSON.stringify(metadata,(key,value) => {
    if (value === undefined || (typeof value === 'number' && !Number.isFinite(value))) {
      throw new Error('Observation is not JSON-safe');
    }
    return value;
  },2) + '\n';
  return {png,metadata,serialized};
}

async function saveBrowserAcquisition(acquired, folder, options) {
  const {png,metadata,serialized} = buildAcquisition(acquired,options);
  // New leaf only. Existing folders are refused, including partial acquisitions.
  await mkdir(folder,{recursive:false});
  await writeFile(join(folder,metadata.image),png,{flag:'wx'});
  await writeFile(join(folder,'capture.json'),serialized,{flag:'wx'});
  return {manifest:join(folder,'capture.json'),metadata};
}

export {buildAcquisition,saveBrowserAcquisition};
