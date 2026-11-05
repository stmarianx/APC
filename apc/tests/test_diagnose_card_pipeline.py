import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from apc.perception.diagnose_card_pipeline import diagnose_frame
from apc.perception.region_model import RANKS, SUITS


class PipelineDiagnosticTests(unittest.TestCase):
    def test_stage_attribution_and_false_proposal_trace(self):
        boxes = [[x,.1,x+.05,.2] for x in (.1,.3,.5,.7,.9)]
        targets = [{"box_xyxy":box,"identity":identity} for box,identity in zip(boxes,["As","Kh","Qd","Jc"])]
        proposals = [{"box_xyxy":boxes[i],"refinement":"test"} for i in (0,1,3,4)]
        class Verifier:
            def filter(self,image,rows):
                annotated = [dict(p,verifier_score=.1 if i == 1 else .9,verifier_route="test") for i,p in enumerate(rows)]
                return [p for p in annotated if p["verifier_score"] >= .5],[annotated[1]]
        def predict(path,rows):
            rank = torch.zeros(3,len(RANKS)); suit = torch.zeros(3,len(SUITS))
            for i,identity in enumerate(("As","2c","7h")):
                rank[i,RANKS.index(identity[0])] = 1
                suit[i,SUITS.index(identity[1])] = 1
            return {"rank":rank,"suit":suit,"ink_present":torch.ones(3,dtype=torch.bool)}
        reader = SimpleNamespace(locator=SimpleNamespace(locator=SimpleNamespace(propose=lambda _: {"proposals":proposals}),
            verifier=Verifier()),predict_regions=predict)
        image = Path(__file__).resolve().parents[1] / "data/reference/capture-stability-demo-v1/frame.png"
        result = diagnose_frame({"image":image,"targets":targets},reader)
        self.assertEqual([t["stage"] for t in result["targets"]],
            ["correct","verifier_rejection","locator_or_refinement","recognition"])
        self.assertEqual(result["targets"][1]["rejected_matches"][0]["verifier_score"],.1)
        self.assertEqual(len(result["false_proposals"]),1)
        self.assertEqual(result["false_proposals"][0]["candidate"],"7h")
        self.assertEqual(result["false_proposals"][0]["diagnostic_index"],3)
        self.assertTrue(all("diagnostic_index" not in p for p in proposals))


if __name__ == "__main__":
    unittest.main()
