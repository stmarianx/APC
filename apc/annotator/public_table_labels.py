"""Map owned-table visible DOM labels into unapproved annotation suggestions."""
from decimal import Decimal, InvalidOperation
import math


def public_table_suggestions(public, alignment):
    if (not isinstance(public,dict) or public.get("source") != "owned_practice_visible_dom_v1"
            or public.get("units") != "BB" or alignment is None):
        raise ValueError("Visible public table labels require measured image alignment")
    skipped = []
    def bb(value):
        if not isinstance(value,str):
            raise ValueError("Visible BB labels must be strings")
        try:
            number = Decimal(value)
        except InvalidOperation as error:
            raise ValueError("Invalid visible BB label") from error
        if not number.is_finite() or number < 0:
            raise ValueError("Invalid visible BB label")
        return value
    def box(row, name):
        values = row.get("box_xywh")
        if (not isinstance(values,list) or len(values) != 4
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                or any(v < 0 for v in values[:2]) or any(v <= 0 for v in values[2:])):
            raise ValueError("Invalid public region geometry")
        try:
            x1,y1,x2,y2 = alignment.box(*values)
        except ValueError:
            skipped.append(name)
            return None
        return {"x":x1,"y":y1,"width":x2-x1,"height":y2-y1}
    seats = public.get("seats")
    if not isinstance(seats,list) or len(seats) != 2:
        raise ValueError("Owned heads-up table requires two visible seats")
    objects = {"seats":[],"action_buttons":[]}
    stacks, heroes, dealers, used = [],[],[],set()
    for seat in seats:
        if not isinstance(seat,dict):
            raise ValueError("Visible seats must be structured objects")
        number = seat.get("seat_no")
        if type(number) is not int or number not in (1,2) or number in used:
            raise ValueError("Invalid visible seat number")
        used.add(number)
        if type(seat.get("is_hero")) is not bool or type(seat.get("has_dealer_button")) is not bool:
            raise ValueError("Visible seat roles must be booleans")
        name = seat.get("player_name")
        if not isinstance(name,str) or not name.strip():
            raise ValueError("Visible seat name missing")
        stack=bb(seat.get("stack_bb"))
        mapped=box(seat,f"seat:{number}")
        if mapped:
            objects["seats"].append({"seat_no":number,"box":mapped,"occupied":True,"is_hero":seat["is_hero"],
                "has_dealer_button":seat["has_dealer_button"],"player_name":name,"stack_bb":stack,
                "raw_stack_text":stack+" BB","status":"unknown","visibility":"uncertain"})
        stacks.append({"seat_no":number,"stack_bb":stack})
        if seat["is_hero"]: heroes.append(number)
        if seat["has_dealer_button"]: dealers.append(number)
    if len(heroes) != 1 or len(dealers) != 1:
        raise ValueError("Visible hero/dealer labels must be unique")
    pot = public.get("pot",{})
    if not isinstance(pot,dict):
        raise ValueError("Visible pot must be a structured object")
    amount=bb(pot.get("amount_bb"))
    mapped=box(pot,"pot")
    if mapped:
        objects["pot"]={"box":mapped,"amount_bb":amount,"raw_text":pot.get("text",amount),"visibility":"uncertain"}
    buttons = public.get("action_buttons")
    if not isinstance(buttons,list):
        raise ValueError("Visible action buttons must be a list")
    legal=[]
    observed_actions=set()
    for row in buttons:
        if not isinstance(row,dict):
            raise ValueError("Visible action buttons must be structured objects")
        action=row.get("action")
        if action not in {"fold","check","call","bet","raise","all_in"} or type(row.get("enabled")) is not bool:
            raise ValueError("Invalid visible action button")
        if action in observed_actions:
            raise ValueError("Duplicate visible action button")
        observed_actions.add(action)
        mapped=box(row,"action:"+action)
        if mapped:
            objects["action_buttons"].append({"box":mapped,"action":action,"enabled":row["enabled"],
                "raw_text":row.get("text",action),"visibility":"uncertain"})
        if row["enabled"]: legal.append(action)
    street=public.get("street")
    if street not in {"preflop","flop","turn","river"}:
        raise ValueError("Invalid visible street")
    return objects,{"street":street,"hero_seat":heroes[0],"dealer_seat":dealers[0],"pot_bb":amount,
                    "to_call_bb":bb(public.get("to_call_bb")),"seat_stacks_bb":stacks,"legal_actions":legal},skipped
