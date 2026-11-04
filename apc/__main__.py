"""APC local application: inspect a frame or play a virtual training hand."""
import argparse
import json
from pathlib import Path
from secrets import randbits


def practice(seed):
    from apc.deadline import ActionCommand
    from apc.full_hand_table import HeadsUpVirtualHand

    hand = HeadsUpVirtualHand(seed=seed, button_player=0)
    while not hand.terminal:
        state = hand.observation()
        print(f"{state['street']} | Hero {' '.join(state['hero_cards'])} | Board {' '.join(state['board']) or '-'} | Pot {state['pot_bb']} BB")
        buttons = {row['action']: row for row in hand.legal_action_buttons()}
        if state['next_actor'] == 'Hero':
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
        except (ValueError, ArithmeticError) as error:
            print(error)
            continue
    print(json.dumps(transition['completed_hand_feedback'], indent=2))


def main():
    parser = argparse.ArgumentParser(prog='APC', description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('desktop', help='Open the APC model controller')
    modes.add_parser('study', help='Search course evidence').add_argument('query')
    modes.add_parser('practice', help='Play heads-up against a check/call opponent').add_argument('--seed', type=int)
    inspect = modes.add_parser('inspect', help='Read card candidates from a saved table frame')
    inspect.add_argument('image', type=Path)
    inspect.add_argument('--corner-symbols', action='store_true', help='Read separate rank and suit glyphs on light corner-indexed cards')
    args = parser.parse_args()
    if args.mode == 'desktop':
        from apc.desktop import launch
        launch()
    elif args.mode == 'study':
        from apc.neural.course_encoder import study
        print(json.dumps(study(args.query), indent=2))
    elif args.mode == 'practice':
        practice(args.seed if args.seed is not None else randbits(32))
    else:
        import torch
        from apc.perception.infer_cards import CardReader
        torch.set_num_threads(1)
        filename = 'corner-glyph-pair-v1.pt' if args.corner_symbols else 'glyph-v1.pt'
        checkpoint = Path(__file__).parent / 'checkpoints' / filename
        print(json.dumps(CardReader(checkpoint, corner_pairs=args.corner_symbols).read(args.image), indent=2))


if __name__ == '__main__':
    main()
