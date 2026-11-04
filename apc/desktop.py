"""Desktop controls for APC models and virtual-chip practice."""
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from secrets import randbits
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from apc.deadline import ActionCommand
from apc.full_hand_table import HeadsUpVirtualHand


def launch():
    root = tk.Tk()
    root.title("APC · AI Poker Coach")
    root.geometry("850x620")
    toolbar = ttk.Frame(root, padding=8)
    toolbar.pack(fill="x")
    query = ttk.Entry(toolbar)
    query.insert(0, "pot odds")
    query.pack(side="left", fill="x", expand=True)
    output = ScrolledText(root, wrap="word", padx=12, pady=12)
    actions = ttk.Frame(root, padding=8)
    actions.pack(side="bottom", fill="x")
    output.pack(fill="both", expand=True)
    pool = ThreadPoolExecutor(max_workers=1)
    hand = None
    events = []
    session = "desktop-" + uuid.uuid4().hex
    generation = 0

    def show(value):
        output.delete("1.0", "end")
        output.insert("1.0", value)

    def clear():
        nonlocal generation
        generation += 1
        for widget in actions.winfo_children():
            widget.destroy()

    def run(job):
        clear()
        ticket = generation
        show("Reading with APC…")
        future = pool.submit(job)
        def poll():
            if ticket != generation:
                return
            if not future.done():
                root.after(80, poll)
                return
            try:
                show(future.result())
            except Exception as error:
                show(str(error))
        poll()

    def lesson():
        text = query.get().strip()
        if text:
            def lookup():
                from apc.neural.course_encoder import study
                return "\n\n".join(r["passage"] + "\n" + json.dumps(r["source"], ensure_ascii=False) for r in study(text))
            run(lookup)

    def inspect():
        path = filedialog.askopenfilename(filetypes=[("Table frames", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            def read():
                from apc.perception.infer_cards import CardReader
                reader = CardReader(Path(__file__).parent / "checkpoints/glyph-v1.pt")
                return json.dumps(reader.read(path), indent=2)
            run(read)

    def display_hand():
        clear()
        state = hand.observation()
        show(json.dumps(state, indent=2))
        size = ttk.Entry(actions, width=8)
        ttk.Label(actions, text="Bet/raise total (BB)").pack(side="left")
        size.pack(side="left")
        def act(action):
            try:
                amount = size.get().strip() if action in ("bet", "raise") else None
                command = ActionCommand(action, to_amount_bb=amount)
                observed = time.monotonic_ns() // 1_000_000
                result = hand.step(command)
                events.append({"observed_monotonic_ms": observed,
                               "state_fingerprint": state["state_fingerprint"],
                               "legal_action_keys": [b["action"] for b in state["action_buttons"]],
                               "chosen_action_key": action, "chosen_action": result["command"],
                               "canonical_state": state})
                while not hand.terminal and hand.observation()["next_actor"] != "Hero":
                    legal = {b["action"] for b in hand.legal_action_buttons()}
                    result = hand.step(ActionCommand("check" if "check" in legal else "call"))
                if hand.terminal:
                    clear()
                    from apc.neural.replay_buffer import APCReplayBuffer
                    from apc.neural.self_play_replay import completed_replay
                    feedback = result["completed_hand_feedback"]
                    buffer = APCReplayBuffer(Path(__file__).parent / "runs" / "desktop-replay" / session)
                    saved = buffer.ingest(completed_replay(hand, list(events), feedback, session))
                    show(json.dumps({"result": feedback, "replay": saved}, indent=2))
                else:
                    display_hand()
            except (ValueError, ArithmeticError) as error:
                show(str(error) + "\n" + json.dumps(hand.observation(), indent=2))
        for button in state["action_buttons"]:
            action = button["action"]
            ttk.Button(actions, text=action, command=lambda a=action: act(a)).pack(side="left")

    def practice():
        nonlocal hand
        events.clear()
        hand = HeadsUpVirtualHand(seed=randbits(32), button_player=0)
        display_hand()

    for label, callback in (("Find lesson", lesson), ("Inspect frame", inspect), ("New practice hand", practice)):
        ttk.Button(toolbar, text=label, command=callback).pack(side="left")
    show("Search the Negreanu curriculum, inspect a saved frame, or play a virtual-chip hand against a check/call opponent.")
    try:
        root.mainloop()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
