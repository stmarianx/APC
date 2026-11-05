# Real browser capture stability demonstration

Captured from APC's own virtual-chip practice table using `practice_browser_adapter.mjs` through the authorized in-app browser on 2026-11-05. No poker action was taken. Visible hero cards: 4h/2s; opponent cards remained hidden. Only DOM text, rectangles, viewport and scroll were read, not internal game state.

The adapter bracketed the PNG with matching observations. PNG size 1265x712, DOM viewport 1280x720, scroll [0,0]. Width-based uniform mapping places card tops approximately at image (519.23,530.12) and (588.41,530.12), consistent with the saved image. This is a developer sanity check, not independent annotation approval. The manifest requires stability validation and binds the PNG hash.

This single frame demonstrates real adapter acquisition plus collection-loader integration. It is not a model accuracy audit, cross-platform or coaching test. No predictions selected the frame. Excluded from training; no readiness credit. Endpoint equality cannot rule out transient movement. Local capture server stopped afterwards; no viewport override used.

The Python callback collector and JS adapter share the evidence contract but are separate runtime implementations. The adapter returns PNG/before/after; here the PNG was saved directly and returned metadata transcribed into collection.json. Automatic cross-runtime persistence/import is not yet implemented.
