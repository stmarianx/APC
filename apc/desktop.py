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
            clear()
            show("Choose the card design in this saved frame. Reads are experimental and are not playing advice.")
            for label, filename in (("Centered text", "glyph-v1.pt"), ("Corner symbols", "corner-glyph-pair-v2.pt"),
                                    ("Corner symbols · mixed-font trial", "corner-glyph-mixed-v1.pt")):
                ttk.Button(actions, text=label, command=lambda name=filename: run(lambda: inspect_saved_frame(path, name))).pack(side="left")

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

    def river_drill():
        clear()
        show("River exercise\n\nThe pot is 10 BB. The bettor has value or air with equal probability. "
             "Value always wins at showdown and air always loses. The defender has a bluff-catcher.\n\n"
             "Choose a bet size. Before revealing the solution, estimate how often air should bet "
             "and how often the defender should call.")
        for wager in (5, 10):
            ttk.Button(actions, text=f"Solve {wager} BB bet",
                       command=lambda size=wager: run(lambda: river_solution_text(size))).pack(side="left")

    def checkpoint_status():
        path = filedialog.askopenfilename(title="Choose APC checkpoint", filetypes=[("Checkpoint", "*.json")])
        if path:
            run(lambda: checkpoint_status_text(path))

    for label, callback in (("Find lesson", lesson), ("Inspect frame", inspect), ("New practice hand", practice),
                            ("River drill", river_drill), ("Model status", checkpoint_status)):
        ttk.Button(toolbar, text=label, command=callback).pack(side="left")
    show("Search the Negreanu curriculum, inspect a saved frame, or play a virtual-chip hand against a check/call opponent.")
    try:
        root.mainloop()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def inspect_saved_frame(path, filename):
    from apc.perception.infer_cards import CardReader
    if filename not in {"glyph-v1.pt", "corner-glyph-pair-v2.pt", "corner-glyph-mixed-v1.pt"}:
        raise ValueError("Unknown bundled card reader")
    reader = CardReader(Path(__file__).parent / "checkpoints" / filename, corner_pairs=filename != "glyph-v1.pt")
    result = reader.read(path)
    return json.dumps({"checkpoint": filename, "inspection": result}, indent=2)


def checkpoint_status_text(path):
    from apc.neural.train_candidate import validate_checkpoint
    from apc.neural.continual_training import validate_completed_replay_checkpoint
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    family = record.get("status")
    if family == "offline_completed_replay_candidate_unpromoted":
        validation = validate_completed_replay_checkpoint(path)
    elif family == "offline_neural_candidate_unpromoted":
        validation = validate_checkpoint(path)
    else:
        raise ValueError("Unsupported checkpoint type; no readiness claim can be made")
    return json.dumps({
        "checkpoint_path": str(Path(path).resolve()),
        "validation": validation,
        "status": family,
        "gates": record.get("gates", {}),
        "latency": record.get("latency", {}),
        "limitations": record.get("limitations", []),
        "notice": "Inspection only. Valid evidence does not activate weights or establish full-table readiness.",
    }, indent=2)


def river_solution_text(bet_bb):
    from coach.src.poker_coach.river_solver import RiverCFRSolver, solve_akq_river
    from apc.neural.river_policy import predict
    game, solution = solve_akq_river(pot_bb=10, bet_bb=bet_bb, iterations=40000)
    bluff = solution.ip_strategy["Air"]["bet"]
    call = solution.oop_strategy["Bluff-catcher"]["call"]
    neural_value, neural_bluff, neural_call = predict(10, bet_bb)
    neural_gap = RiverCFRSolver.policy_exploitability_bb(game, [neural_value, neural_bluff], [neural_call])
    return (f"River solution · 10 BB pot · {bet_bb} BB bet\n\n"
            f"Bet with air {bluff:.1%} of the time. Call with the bluff-catcher {call:.1%} of the time.\n"
            f"Neural policy estimate: bluff {neural_bluff:.1%}, bluff-catcher call {neural_call:.1%}.\n"
            f"A bluff risks {bet_bb} BB to win 10 BB. A call risks {bet_bb} BB to win {10 + bet_bb} BB.\n\n"
            f"Source: APC river CFR solver, 40,000 iterations, equal prior weights for value and air.\n"
            f"Solver exploitability: {solution.exploitability_bb:.4f} BB.\n"
            f"Neural policy exploitability: {neural_gap:.4f} BB.")
