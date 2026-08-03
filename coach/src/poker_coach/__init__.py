"""Offline poker coaching domain core."""

from .models import (
    ActionKind,
    Card,
    HandAction,
    HandHistory,
    HoleCards,
    Player,
    PotAward,
    Street,
)
from .features import ProfileBook, position_map
from .equity import (
    EquityResult,
    HandRank,
    WeightedCombo,
    WeightedEquityResult,
    best_hand_rank,
    equity_vs_hand,
    equity_vs_range,
)
from .pokerstars import PokerStarsParseError, PokerStarsParser
from .replay import DecisionSnapshot, HandReplayer, PlayerLedger, ReplayResult
from .report import analyze_hands
from .ranges import expand_class, parse_range
from .solutions import ActionSolution, InMemorySolutionStore, SolutionKey, SolvedSpot
from .explanations import action_id, explain_decision
from .storage import CoachDatabase, ImportResult, IngestedFileState
from .ingest import FolderScanResult, HandHistoryFolderScanner, split_completed_hands
from .trainer import (
    ScenarioLibrary,
    StrategyProvenance,
    TrainingAction,
    TrainingScenario,
    TrainingService,
    TrainingSession,
)
from .river_solver import HandBucket, RiverCFRSolver, RiverSolution, RiverSubgame, solve_akq_river
from .solver_import import (
    SolverBundle,
    SolverBundleError,
    SolverBundleImporter,
    SolverImportResult,
    solved_spot_to_dict,
)
from .solver_adapters import (
    BUNDLE_JSON_V1,
    TABULAR_CSV_V1,
    ParsedSolverExport,
    SolverExportRegistry,
    TabularSolverCSVAdapter,
)
from .matching import (
    DecisionContext,
    DecisionSolutionMatch,
    DecisionSolutionMatcher,
    analyze_with_solutions,
    build_drill_queue,
)
from .study import (
    InMemoryStudyStore,
    ReviewRating,
    StudyState,
    StudyStore,
    schedule_review,
)
from .isomorphism import canonicalize_suit_state, suit_isomorphic
from .solution_tree import SolutionForest, SolutionTreeNode
from .range_strategy import RANKS, aggregate_range_strategies, hand_class, public_node_fingerprint
from .exploit import ExploitRule, aggregate_posterior, player_exploit_insights
from .solver_practice import SolverPracticeService, SolverPracticeSession
from .live_state import LiveTableService, LiveTableSession, LiveTableState
from .live_capture import (
    LiveCapturePoll,
    PokerStarsDecisionProjector,
    PokerStarsLiveTailAdapter,
    ProjectedDecision,
)
from .visual_capture import (
    NormalizedBox,
    VisualField,
    VisualObservation,
    VisualObservationAdapter,
)
from .board_texture import BoardTexture, RANGE_CAVEAT, analyze_board_texture
from .range_matchup import RangeMatchupResult, analyze_range_matchup
from .range_inference import combo_label, condition_solution_range
from .range_timeline import (
    build_opponent_range_timelines,
    opponent_decision_contexts,
)
from .range_calibration import score_opponent_range_timelines
from .state_transition import StateTransitionError, validate_state_transition
from .strategy_selection import (
    DEFAULT_LATENCY_BUDGET_MS,
    SAFE_CERTIFICATE_METHOD,
    RefinementSafetyCertificate,
    StrategySelectionService,
    SubgameRefinement,
    SubgameRefinementRequest,
    SubgameRefiner,
)
from .trends import DEFAULT_SESSION_GAP, analyze_hero_trends

__all__ = [
    "ActionKind",
    "ActionSolution",
    "Card",
    "CoachDatabase",
    "HandAction",
    "HandHistory",
    "HandRank",
    "HandReplayer",
    "InMemorySolutionStore",
    "ImportResult",
    "IngestedFileState",
    "FolderScanResult",
    "HandHistoryFolderScanner",
    "HoleCards",
    "Player",
    "PlayerLedger",
    "ProfileBook",
    "EquityResult",
    "PokerStarsParseError",
    "PokerStarsParser",
    "PotAward",
    "ReplayResult",
    "SolutionKey",
    "SolvedSpot",
    "Street",
    "ScenarioLibrary",
    "StrategyProvenance",
    "TrainingAction",
    "TrainingScenario",
    "TrainingService",
    "TrainingSession",
    "HandBucket",
    "RiverCFRSolver",
    "RiverSolution",
    "RiverSubgame",
    "solve_akq_river",
    "SolverBundle",
    "SolverBundleError",
    "SolverBundleImporter",
    "SolverImportResult",
    "solved_spot_to_dict",
    "BUNDLE_JSON_V1",
    "TABULAR_CSV_V1",
    "ParsedSolverExport",
    "SolverExportRegistry",
    "TabularSolverCSVAdapter",
    "DecisionContext",
    "DecisionSolutionMatch",
    "DecisionSolutionMatcher",
    "analyze_with_solutions",
    "build_drill_queue",
    "InMemoryStudyStore",
    "ReviewRating",
    "StudyState",
    "StudyStore",
    "schedule_review",
    "canonicalize_suit_state",
    "suit_isomorphic",
    "SolutionForest",
    "SolutionTreeNode",
    "RANKS",
    "aggregate_range_strategies",
    "hand_class",
    "public_node_fingerprint",
    "ExploitRule",
    "aggregate_posterior",
    "player_exploit_insights",
    "SolverPracticeService",
    "SolverPracticeSession",
    "LiveTableService",
    "LiveTableSession",
    "LiveTableState",
    "LiveCapturePoll",
    "PokerStarsDecisionProjector",
    "PokerStarsLiveTailAdapter",
    "ProjectedDecision",
    "NormalizedBox",
    "VisualField",
    "VisualObservation",
    "VisualObservationAdapter",
    "BoardTexture",
    "RANGE_CAVEAT",
    "analyze_board_texture",
    "RangeMatchupResult",
    "analyze_range_matchup",
    "combo_label",
    "condition_solution_range",
    "build_opponent_range_timelines",
    "opponent_decision_contexts",
    "score_opponent_range_timelines",
    "StateTransitionError",
    "validate_state_transition",
    "DEFAULT_LATENCY_BUDGET_MS",
    "SAFE_CERTIFICATE_METHOD",
    "RefinementSafetyCertificate",
    "StrategySelectionService",
    "SubgameRefinement",
    "SubgameRefinementRequest",
    "SubgameRefiner",
    "DEFAULT_SESSION_GAP",
    "analyze_hero_trends",
    "WeightedCombo",
    "WeightedEquityResult",
    "DecisionSnapshot",
    "best_hand_rank",
    "equity_vs_hand",
    "equity_vs_range",
    "expand_class",
    "parse_range",
    "position_map",
    "split_completed_hands",
    "analyze_hands",
    "action_id",
    "explain_decision",
]
