"""APC local application: inspect a frame or play a virtual training hand."""
import argparse
import json
import time
from pathlib import Path
from secrets import randbits


def practice(seed, *, replay_dir=None, session_id=None):
    from apc.deadline import ActionCommand
    from apc.full_hand_table import HeadsUpVirtualHand

    hand = HeadsUpVirtualHand(seed=seed, button_player=0)
    events = []
    while not hand.terminal:
        state = hand.observation()
        print(f"{state['street']} | Hero {' '.join(state['hero_cards'])} | Board {' '.join(state['board']) or '-'} | Pot {state['pot_bb']} BB")
        buttons = {row['action']: row for row in hand.legal_action_buttons()}
        if state['next_actor'] == 'Hero':
            observed_ms = time.monotonic_ns() // 1_000_000
            print(f"Stacks {state['stacks_bb']} BB | To call {state['to_call_bb']} BB")
            for action, button in buttons.items():
                print(action, button.get('minimum_to_bb', ''))
            try:
                parts = input('Action [total size in BB], or quit > ').split()
            except EOFError:
                return
            if parts == ['quit']:
                return
            if not parts or parts[0] not in buttons or len(parts) > 2:
                continue
            command = ActionCommand(parts[0], to_amount_bb=parts[1] if len(parts) == 2 else None)
        else:
            action = 'check' if 'check' in buttons else 'call'
            command = ActionCommand(action)
            print('Practice opponent:', action)
        try:
            transition = hand.step(command)
            if replay_dir is not None and state['next_actor'] == 'Hero':
                events.append({"observed_monotonic_ms": observed_ms,
                               "state_fingerprint": state["state_fingerprint"],
                               "legal_action_keys": list(buttons), "chosen_action_key": parts[0],
                               "chosen_action": transition["command"], "canonical_state": state})
        except (ValueError, ArithmeticError) as error:
            print(error)
            continue
    print(json.dumps(transition['completed_hand_feedback'], indent=2))
    if replay_dir is not None:
        from apc.neural.replay_buffer import APCReplayBuffer
        from apc.neural.self_play_replay import completed_replay
        replay = completed_replay(hand, events, transition["completed_hand_feedback"], session_id)
        print(json.dumps(APCReplayBuffer(replay_dir).ingest(replay), indent=2))


def main():
    parser = argparse.ArgumentParser(prog='APC', description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('desktop', help='Open the APC model controller')
    modes.add_parser('study', help='Search course evidence').add_argument('query')
    table = modes.add_parser('practice', help='Play heads-up against a check/call opponent')
    table.add_argument('--seed', type=int)
    table.add_argument('--replay-dir', type=Path, help='Save completed practice hands for offline learning')
    table.add_argument('--session-id', help='Reuse this ID across hands in the same recorded session')
    inspect = modes.add_parser('inspect', help='Read card candidates from a saved table frame')
    inspect.add_argument('image', type=Path)
    inspect.add_argument('--corner-symbols', action='store_true', help='Read separate rank and suit glyphs on light corner-indexed cards')
    inspect.add_argument('--neural-cards', action='store_true', help='Experimental neural localization plus surface-refined recognition; no coaching readiness')
    inspect.add_argument('--adaptive-glyphs', action='store_true', help='Opt-in full-face/corner glyph extraction; requires --neural-cards')
    inspect.add_argument('--glyph-checkpoint', type=Path, help='Explicit experimental glyph weights; requires --neural-cards')
    args = parser.parse_args()
    if args.mode == 'desktop':
        from apc.desktop import launch
        launch()
    elif args.mode == 'study':
        from apc.neural.course_encoder import study
        print(json.dumps(study(args.query), indent=2))
    elif args.mode == 'practice':
        if (args.replay_dir is None) != (args.session_id is None) or (args.session_id is not None and not args.session_id.strip()):
            parser.error('Recording requires both --replay-dir and a nonempty --session-id')
        practice(args.seed if args.seed is not None else randbits(32), replay_dir=args.replay_dir, session_id=args.session_id)
    else:
        if args.adaptive_glyphs and not args.neural_cards:
            parser.error('--adaptive-glyphs requires --neural-cards')
        if args.glyph_checkpoint and not args.neural_cards:
            parser.error('--glyph-checkpoint requires --neural-cards')
        import torch
        from apc.perception.infer_cards import CardReader
        torch.set_num_threads(1)
        filename = 'glyph-surface-pair-v1.pt' if args.neural_cards else ('corner-glyph-pair-v2.pt' if args.corner_symbols else 'glyph-v1.pt')
        checkpoint = args.glyph_checkpoint or Path(__file__).parent / 'checkpoints' / filename
        options = {"locator_checkpoint": Path(__file__).parent / "checkpoints/card-locator-mixed-v1.pt",
                   "refine_surfaces": True} if args.neural_cards else {}
        if args.adaptive_glyphs:
            options['surface_pairs'] = True
        print(json.dumps(CardReader(checkpoint, corner_pairs=args.corner_symbols or args.neural_cards, **options).read(args.image), indent=2))


if __name__ == '__main__':
    main()
