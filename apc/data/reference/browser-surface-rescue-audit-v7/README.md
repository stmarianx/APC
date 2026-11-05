# Audit v7 aborted before scoring

The first scheduled acquisition was rejected by the frozen saver as `Card outside saved screenshot`. Both DOM observations match at viewport 920x900, but the saved image is 905x509. Under the assumed uniform-width mapping its vertical extent is only about 517 DOM pixels, while hero cards start at y=536.40625. The capture is ineligible and preserved unchanged.

Important visual finding: the saved image DOES show the cards around image y=379, with page content occupying only roughly 650 of the 905 image pixels horizontally. Thus this is not established physical clipping: the raster uses a different scale/canvas from the DOM viewport. A stale/mismatched rendered viewport following resize is suspected, not proven. Equal before/after DOM metadata cannot establish raster synchronization. Do not repair this frame by guessing a scale or score it as a model miss.

The preregistered rejection policy was honored: one acquisition retained, zero accepted frames, zero retakes, zero scoring, zero training. No collection.json was assembled. status.json records the frozen plan hash and outcome. No model/default changes or readiness credit. Browser viewport override was reset and local server session 3377 stopped.

Next: investigate resize/raster synchronization using separately declared setup captures and image-space landmarks, then preregister a NEW audit with an explicit capture preflight policy. Do not resume v7 or silently replace its failed acquisition. The APC project goal remains active; only this audit schedule is aborted.
