"""Deterministic negative scenes for capture audits, not gameplay transitions."""

SCENES = {
    "text": '<section class="panel"><h2>Player panel</h2><p>Stack 100 BB</p><p>Player 7</p><p>Pot 0 BB</p></section><aside>Call 2.5 BB<br>Raise 30 BB<br>Check / Fold</aside>',
    "lobby": '<h2>Waiting for seats</h2><div class="lobby"><section>Seat 1<br>100 BB</section><section>Seat 2<br>80 BB</section><section>Seat 3<br>120 BB</section></div><p>No hand in progress</p>',
    "controls": '<div class="controls"><button disabled>Fold</button><button disabled>Check</button><button disabled>Call 2.5 BB</button><button disabled>Raise</button></div><div class="tiles"><span>OK</span><span>30</span><span>BB</span></div><p>Controls disabled · Pot 0 BB</p>',
}


def render_negative_fixture(scene):
    if scene not in SCENES:
        raise ValueError("Unknown negative scene")
    return ('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>APC audit negative fixture</title><style>
body{background:#10231d;color:#eff6f2;font-family:system-ui;margin:24px}h1{font-size:24px}
main{min-height:580px;border:8px solid #4a4438;border-radius:36px;padding:24px;background:#21543f}
.panel{background:#e5e4d7;color:#183126;padding:16px;width:180px;margin-top:120px}
aside{margin:24px;padding:16px;background:#cae791;color:#183126;width:120px}
.lobby{display:flex;flex-wrap:wrap;gap:24px;margin-top:100px}.lobby section{background:#e5e4d7;color:#183126;padding:24px}
.controls{display:flex;flex-wrap:wrap;gap:12px;margin-top:180px}button{font:inherit;padding:16px;background:#e5e4d7;color:#183126}
.tiles{display:flex;gap:10px;margin-top:32px}.tiles span{background:#f5f3e9;color:#183126;width:36px;height:72px;padding:4px}
</style><h1>APC · negative capture fixture</h1><p>No cards are present. Virtual chips only.</p>
<main data-apc-audit-negative="''' + scene + '">' + SCENES[scene] +
        '</main><p>Static controlled fixture, not an observed gameplay transition.</p></html>').encode("utf-8")
