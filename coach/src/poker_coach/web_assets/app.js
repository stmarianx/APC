"use strict";

const state = { session: null, feedback: null, rangeGroups: [], rangeWeights: new Map(), rangeInference: null, solverPractice: null, liveSession: null, liveResult: null };
let scanTimer = null;
let scanInProgress = false;
let livePollTimer = null;
let livePollInProgress = false;
const $ = (id) => document.getElementById(id);
const suitMap = { c: "♣", d: "♦", h: "♥", s: "♠" };

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function cardNode(token, placeholder = false) {
  const node = document.createElement("div");
  node.className = `card${placeholder ? " placeholder" : ""}`;
  if (!placeholder) {
    const rank = token.slice(0, -1);
    const suit = token.slice(-1);
    if (suit === "h" || suit === "d") node.classList.add("red");
    const rankNode = document.createElement("span"); rankNode.textContent = rank;
    const suitNode = document.createElement("span"); suitNode.className = "suit"; suitNode.textContent = suitMap[suit];
    node.append(rankNode, suitNode);
  }
  return node;
}

function renderCards(container, cards, targetCount) {
  container.replaceChildren();
  cards.forEach((token) => container.appendChild(cardNode(token)));
  for (let index = cards.length; index < targetCount; index += 1) container.appendChild(cardNode("", true));
}

function renderScenario(session) {
  state.session = session;
  state.feedback = null;
  const scenario = session.scenario;
  if (!scenario) return renderComplete(session);
  $("feedbackPanel").classList.add("hidden");
  $("scenarioTitle").textContent = scenario.title;
  $("streetBadge").textContent = scenario.street;
  $("difficultyBadge").textContent = scenario.difficulty;
  $("heroPosition").textContent = `Hero · ${scenario.hero_position}`;
  $("villainPosition").textContent = `Villain · ${scenario.villain_position}`;
  $("heroStack").textContent = `${scenario.effective_stack_bb.toFixed(1)}bb effective`;
  $("villainStack").textContent = `${scenario.effective_stack_bb.toFixed(1)}bb effective`;
  $("potValue").textContent = `${scenario.pot_bb.toFixed(1)}bb`;
  $("callPrice").textContent = scenario.to_call_bb > 0 ? `${scenario.to_call_bb.toFixed(1)}bb to call` : "Checked to you";
  $("decisionPrompt").textContent = scenario.to_call_bb > 0 ? "Continue or release?" : "Choose your line";
  renderCards($("heroCards"), scenario.hero_cards, 2);
  renderCards($("board"), scenario.board, scenario.board.length === 0 ? 0 : 5);
  $("actionHistory").replaceChildren(...scenario.action_history.map((text) => { const li = document.createElement("li"); li.textContent = text; return li; }));
  $("concepts").replaceChildren(...scenario.concepts.map((text) => { const chip = document.createElement("span"); chip.textContent = text; return chip; }));
  $("provenanceTier").textContent = scenario.provenance.solver_verified ? "Solver verified" : "Educational baseline";
  $("provenanceText").textContent = `${scenario.provenance.source} · ${scenario.provenance.version}`;
  updateProgress(session.progress);
  const buttons = scenario.actions.map((action) => {
    const button = document.createElement("button");
    button.className = "action-button";
    button.textContent = action.label;
    button.addEventListener("click", () => submitDecision(action.action_id));
    return button;
  });
  $("actionButtons").replaceChildren(...buttons);
}

function updateProgress(progress) {
  $("progressText").textContent = `${progress.answered} / ${progress.total}`;
  $("progressFill").style.width = `${progress.total ? (100 * progress.answered / progress.total) : 0}%`;
  $("scoreValue").textContent = progress.score;
}

async function startSession() {
  try {
    const session = await api("/api/sessions", { method: "POST", body: JSON.stringify({ count: 6 }) });
    renderScenario(session);
  } catch (error) { showFatal(error); }
}

async function submitDecision(actionId) {
  document.querySelectorAll(".action-button").forEach((button) => { button.disabled = true; });
  try {
    const result = await api(`/api/sessions/${state.session.session_id}/decisions`, { method: "POST", body: JSON.stringify({ action_id: actionId }) });
    state.feedback = result.feedback;
    updateProgress(result.progress);
    renderFeedback(result.feedback, result.complete);
  } catch (error) {
    document.querySelectorAll(".action-button").forEach((button) => { button.disabled = false; });
    showFatal(error);
  }
}

function pct(value) { return value == null ? "—" : `${(value * 100).toFixed(1)}%`; }

function renderFeedback(feedback, complete) {
  const scenario = feedback.scenario;
  $("feedbackPanel").classList.remove("hidden");
  $("feedbackTitle").textContent = feedback.grade === "major_leak" ? "Rebuild this spot" : "Decision reviewed";
  $("gradeBadge").className = `grade-badge ${feedback.grade}`;
  $("gradeBadge").textContent = feedback.grade.replace("_", " ");
  $("feedbackMessage").textContent = feedback.message;
  $("evLoss").textContent = `${feedback.ev_loss_bb.toFixed(2)}bb`;
  $("chosenFrequency").textContent = pct(feedback.chosen_frequency);
  $("requiredEquity").textContent = pct(scenario.math.break_even_call_equity);
  $("sprValue").textContent = scenario.math.spr.toFixed(1);
  $("explanation").textContent = scenario.explanation;
  const bars = scenario.actions.map((action) => {
    const row = document.createElement("div"); row.className = "strategy-row";
    const label = document.createElement("strong"); label.textContent = action.label;
    const track = document.createElement("div"); track.className = "bar-track";
    const fill = document.createElement("span"); fill.style.width = `${action.frequency * 100}%`; track.appendChild(fill);
    const frequency = document.createElement("small"); frequency.textContent = pct(action.frequency);
    const ev = document.createElement("small"); ev.textContent = `${action.ev_bb.toFixed(2)} EV`;
    row.append(label, track, frequency, ev); return row;
  });
  $("strategyBars").replaceChildren(...bars);
  $("nextButton").textContent = complete ? "Start another session" : "Next decision";
  $("nextButton").onclick = complete ? startSession : nextDecision;
  $("feedbackPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function nextDecision() {
  try {
    const session = await api(`/api/sessions/${state.session.session_id}`);
    renderScenario(session);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) { showFatal(error); }
}

function renderComplete(session) {
  updateProgress(session.progress);
  $("scenarioTitle").textContent = "Session complete";
}

function solverBar(labelText, actionText, frequency) {
  const row = document.createElement("div"); row.className = "solver-strategy-row";
  const labels = document.createElement("div");
  const label = document.createElement("strong"); label.textContent = labelText;
  const action = document.createElement("small"); action.textContent = actionText;
  labels.append(label, action);
  const track = document.createElement("div"); track.className = "bar-track";
  const fill = document.createElement("span"); fill.style.width = `${frequency * 100}%`; track.appendChild(fill);
  const value = document.createElement("strong"); value.textContent = pct(frequency);
  row.append(labels, track, value); return row;
}

async function solveRiver() {
  $("solverError").classList.add("hidden");
  $("solveRiverButton").disabled = true;
  $("solveRiverButton").textContent = "Solving…";
  try {
    const pot = Number($("solverPot").value);
    const bet = Number($("solverBet").value);
    const iterations = Number($("solverIterations").value);
    const result = await api("/api/solve-river", { method: "POST", body: JSON.stringify({ pot_bb: pot, bet_bb: bet, iterations }) });
    renderSolverResult(result);
  } catch (error) {
    $("solverError").textContent = error.message;
    $("solverError").classList.remove("hidden");
  } finally {
    $("solveRiverButton").disabled = false;
    $("solveRiverButton").textContent = "Run CFR solver";
  }
}

function renderSolverResult(result) {
  $("solverResults").classList.remove("hidden");
  $("solverExploitability").textContent = `${result.exploitability_bb.toFixed(4)}bb`;
  $("solverNashGap").textContent = `${result.nash_gap_bb.toFixed(4)}bb`;
  $("solverGameEv").textContent = `${result.expected_ip_ev_bb.toFixed(3)}bb`;
  $("solverIterationResult").textContent = result.iterations.toLocaleString();
  const ip = result.strategy.ip;
  const oop = result.strategy.oop;
  $("ipSolverBars").replaceChildren(
    solverBar("Value", "Bet", ip.Value.bet),
    solverBar("Air", "Bluff", ip.Air.bet),
  );
  $("oopSolverBars").replaceChildren(
    solverBar("Bluff-catcher", "Call", oop["Bluff-catcher"].call),
  );
  const bluffShare = ip.Air.bet / (ip.Value.bet + ip.Air.bet);
  $("solverInterpretation").textContent = `At this ${result.game.bet_bb.toFixed(1)}bb bet into ${result.game.pot_bb.toFixed(1)}bb, Air bluffs ${pct(ip.Air.bet)} of the time. Bluffs make up ${pct(bluffShare)} of IP's betting range, while the bluff-catcher defends ${pct(oop["Bluff-catcher"].call)}. The remaining Nash gap measures approximation error.`;
  $("solverResults").scrollIntoView({behavior:"smooth", block:"nearest"});
}

function rangeDistributionRows(distribution) {
  return Object.entries(distribution).map(([category, frequency]) =>
    solverBar(category.replaceAll("_", " "), "Current made-hand share", Number(frequency))
  );
}

async function analyzeRangeMatchup() {
  $("rangeMatchupError").classList.add("hidden");
  $("analyzeRangeMatchupButton").disabled = true;
  $("analyzeRangeMatchupButton").textContent = "Analyzingâ€¦";
  try {
    const board = $("rangeMatchupBoard").value.trim().split(/[\s,]+/).filter(Boolean);
    const result = await api("/api/analyze-range-matchup", {
      method: "POST",
      body: JSON.stringify({
        board,
        hero_range: $("rangeMatchupHero").value.trim(),
        villain_range: $("rangeMatchupVillain").value.trim(),
        samples: Number($("rangeMatchupSamples").value),
        seed: Number($("rangeMatchupSeed").value),
      }),
    });
    $("rangeMatchupResults").classList.remove("hidden");
    $("rangeHeroEquity").textContent = pct(Number(result.equity.hero));
    $("rangeVillainEquity").textContent = pct(Number(result.equity.villain));
    const edge = Number(result.equity.hero_edge);
    $("rangeEquityEdge").textContent = `${edge >= 0 ? "+" : ""}${(edge * 100).toFixed(1)}pp`;
    $("rangeMethod").textContent = result.method.replaceAll("_", " ");
    $("rangeOutcomes").textContent = Number(result.outcomes_evaluated).toLocaleString();
    $("rangeConfidence").textContent = `${pct(Number(result.equity.confidence_95.lower))} - ${pct(Number(result.equity.confidence_95.upper))}`;
    $("rangeHeroDistribution").replaceChildren(...rangeDistributionRows(result.hero_range.category_distribution));
    $("rangeVillainDistribution").replaceChildren(...rangeDistributionRows(result.villain_range.category_distribution));
    const nuts = result.current_range_relative_nuts;
    if (nuts.hero_nut_share == null) {
      $("rangeNutLeader").textContent = "Current nut share unavailable preflop";
      $("rangeNutShares").textContent = "Equity still includes complete sampled runouts.";
    } else {
      $("rangeNutLeader").textContent = `${nuts.leader === "even" ? "Even" : `${nuts.leader} leads`} current nut share`;
      $("rangeNutShares").textContent = `Hero ${pct(Number(nuts.hero_nut_share))} | Villain ${pct(Number(nuts.villain_nut_share))} | strongest ${nuts.strongest_hand.category.replaceAll("_", " ")}`;
    }
    $("rangeMatchupProvenance").textContent = `${result.provenance.weighting}. ${result.provenance.caveat}`;
    $("rangeMatchupResults").scrollIntoView({behavior:"smooth", block:"nearest"});
  } catch (error) {
    $("rangeMatchupError").textContent = error.message;
    $("rangeMatchupError").classList.remove("hidden");
  } finally {
    $("analyzeRangeMatchupButton").disabled = false;
    $("analyzeRangeMatchupButton").textContent = "Analyze ranges";
  }
}

async function loadSolverBundle() {
  $("solverImportError").classList.add("hidden");
  try {
    const format = $("solverExportFormat").value;
    const sample = await api(`/api/sample-solver-export?format=${encodeURIComponent(format)}`);
    $("solverBundleInput").value = sample.content;
  } catch (error) {
    $("solverImportError").textContent = error.message;
    $("solverImportError").classList.remove("hidden");
  }
}

async function importSolverBundle() {
  $("solverImportError").classList.add("hidden");
  try {
    const format = $("solverExportFormat").value;
    const content = $("solverBundleInput").value;
    const result = await api("/api/import-solver-export", {method:"POST", body:JSON.stringify({format, content})});
    $("solverImportResult").classList.remove("hidden");
    $("importedSpotCount").textContent = result.spots;
    $("insertedSpotCount").textContent = result.inserted;
    $("updatedSpotCount").textContent = result.updated;
    $("importedFormat").textContent = result.format.replaceAll("-", " ");
    $("importedTreeEdges").textContent = result.tree.linked_edges;
    $("importedTreeDepth").textContent = result.tree.max_depth;
    $("importedTreeAmbiguous").textContent = result.tree.ambiguous_nodes;
    const treeNodes = new Map(result.tree.node_rows.map((node) => [node.fingerprint, node]));
    const nodes = result.solutions.map((solution, index) => {
      const article = document.createElement("article"); article.className = "solution-node";
      const head = document.createElement("div");
      const name = document.createElement("strong"); name.textContent = result.node_ids[index];
      const fingerprint = document.createElement("small"); fingerprint.textContent = solution.fingerprint.slice(0, 16);
      head.append(name, fingerprint);
      const state = document.createElement("span");
      const tree = treeNodes.get(solution.fingerprint);
      state.textContent = `${tree.street} · depth ${tree.depth} · ${solution.key.hero_position} · ${solution.key.hero_cards.join(" ")} · ${solution.key.board.join(" ") || "preflop"}`;
      const actions = document.createElement("span");
      actions.textContent = solution.actions.map((action) => `${action.action} ${pct(Number(action.frequency))}`).join(" · ");
      article.append(head, state, actions); return article;
    });
    $("importedSolutions").replaceChildren(...nodes);
    renderRangeExplorer(result.range_strategies);
  } catch (error) {
    $("solverImportError").textContent = error.message;
    $("solverImportError").classList.remove("hidden");
  }
}

function matrixClass(ranks, row, column) {
  if (row === column) return `${ranks[row]}${ranks[column]}`;
  if (row < column) return `${ranks[row]}${ranks[column]}s`;
  return `${ranks[column]}${ranks[row]}o`;
}

function renderRangeExplorer(payload) {
  state.rangeGroups = payload.groups || [];
  if (!state.rangeGroups.length) return $("rangeExplorer").classList.add("hidden");
  $("rangeExplorer").classList.remove("hidden");
  const options = state.rangeGroups.map((group) => {
    const option = document.createElement("option");
    option.value = group.public_fingerprint;
    option.textContent = `${group.label} · ${group.covered_classes} classes`;
    return option;
  });
  $("rangeNodeSelect").replaceChildren(...options);
  renderSelectedRangeGroup();
}

function renderSelectedRangeGroup() {
  const group = state.rangeGroups.find((row) => row.public_fingerprint === $("rangeNodeSelect").value) || state.rangeGroups[0];
  if (!group) return;
  const ranks = "AKQJT98765432".split("");
  const cells = new Map(group.cells.map((cell) => [cell.hand_class, cell]));
  const buttons = [];
  ranks.forEach((_, row) => ranks.forEach((__, column) => {
    const label = matrixClass(ranks, row, column);
    const cell = cells.get(label);
    const button = document.createElement("button");
    button.className = `range-cell${cell ? " available" : " missing"}`;
    const name = document.createElement("strong"); name.textContent = label;
    const weightText = document.createElement("small");
    if (!cell) {
      button.disabled = true; weightText.textContent = "—";
    } else {
      const key = `${group.public_fingerprint}:${label}`;
      if (!state.rangeWeights.has(key)) state.rangeWeights.set(key, 1);
      const weight = state.rangeWeights.get(key);
      weightText.textContent = weight === 0 ? "off" : `${weight * 100}%`;
      button.classList.toggle("excluded", weight === 0);
      const dominant = [...cell.actions].sort((a,b) => Number(b.frequency) - Number(a.frequency))[0];
      button.style.setProperty("--mix", String(Number(dominant.frequency)));
      button.title = `${cell.samples} exact node(s) · ${dominant.action} ${pct(Number(dominant.frequency))}`;
      button.addEventListener("click", () => {
        const current = state.rangeWeights.get(key);
        state.rangeWeights.set(key, current === 1 ? 0.5 : current === 0.5 ? 0 : 1);
        state.rangeInference = null;
        $("rangeInferenceResults").classList.add("hidden");
        renderSelectedRangeGroup();
      });
    }
    button.append(name, weightText); buttons.push(button);
  }));
  $("rangeMatrix").replaceChildren(...buttons);
  $("rangeExactNodes").textContent = group.private_nodes;
  $("rangeCoveredClasses").textContent = group.covered_classes;
  const totals = new Map(); let totalWeight = 0; let selectedClasses = 0;
  group.cells.forEach((cell) => {
    const weight = state.rangeWeights.get(`${group.public_fingerprint}:${cell.hand_class}`) ?? 1;
    if (weight > 0) selectedClasses += 1;
    const sampleWeight = weight * cell.samples; totalWeight += sampleWeight;
    cell.actions.forEach((action) => totals.set(action.action, (totals.get(action.action) || 0) + sampleWeight * Number(action.frequency)));
  });
  $("rangeSelectionSummary").textContent = `${selectedClasses} of ${group.covered_classes} covered classes included · ${group.source} ${group.source_version}`;
  const mix = [...totals.entries()].map(([action, value]) => solverBar(action, "Weighted frequency", totalWeight ? value / totalWeight : 0));
  $("rangeActionMix").replaceChildren(...mix);
  const actionIds = [...new Set(group.cells.flatMap((cell) => cell.actions.map((action) => action.action)))].sort();
  const priorAction = $("rangeInferenceAction").value;
  const actionOptions = actionIds.map((actionId) => {
    const option = document.createElement("option"); option.value = actionId; option.textContent = actionId.replaceAll("_", " "); return option;
  });
  $("rangeInferenceAction").replaceChildren(...actionOptions);
  if (actionIds.includes(priorAction)) $("rangeInferenceAction").value = priorAction;
  $("conditionRangeButton").disabled = actionIds.length === 0;
  if (state.rangeInference?.public_fingerprint !== group.public_fingerprint) {
    state.rangeInference = null;
    $("rangeInferenceResults").classList.add("hidden");
  }
  if (state.solverPractice?.public_fingerprint !== group.public_fingerprint) {
    state.solverPractice = null;
    $("solverPracticePanel").classList.add("hidden");
  }
}

async function conditionSelectedRange() {
  const group = state.rangeGroups.find((row) => row.public_fingerprint === $("rangeNodeSelect").value) || state.rangeGroups[0];
  if (!group) return;
  $("rangeInferenceError").classList.add("hidden");
  $("conditionRangeButton").disabled = true;
  $("conditionRangeButton").textContent = "Conditioning...";
  const priorWeights = {};
  group.cells.forEach((cell) => {
    const weight = state.rangeWeights.get(`${group.public_fingerprint}:${cell.hand_class}`) ?? 1;
    cell.exact_combos.forEach((combo) => { priorWeights[combo] = weight; });
  });
  try {
    const result = await api(`/api/range-strategies/${group.public_fingerprint}/condition`, {
      method: "POST",
      body: JSON.stringify({observed_action: $("rangeInferenceAction").value, prior_weights: priorWeights}),
    });
    state.rangeInference = result;
    $("rangeInferenceResults").classList.remove("hidden");
    $("rangeActionEvidence").textContent = pct(Number(result.action_probability_under_prior));
    $("rangeEntropyShift").textContent = `${Number(result.information.prior_entropy_bits).toFixed(3)} -> ${Number(result.information.posterior_entropy_bits).toFixed(3)} bits`;
    $("rangeEffectiveShift").textContent = `${Number(result.information.prior_effective_combos).toFixed(2)} -> ${Number(result.information.posterior_effective_combos).toFixed(2)}`;
    $("rangeTotalVariation").textContent = pct(Number(result.information.total_variation_shift));
    const rows = result.combos.map((combo) => {
      const article = document.createElement("article"); article.className = "range-inference-row";
      const identity = document.createElement("div");
      const name = document.createElement("strong"); name.textContent = combo.combo;
      const detail = document.createElement("small"); detail.textContent = `${combo.hand_class} | likelihood ${pct(Number(combo.action_likelihood))}`;
      identity.append(name, detail);
      const track = document.createElement("div"); track.className = "bar-track";
      const fill = document.createElement("span"); fill.style.width = `${Number(combo.posterior) * 100}%`; track.appendChild(fill);
      const shift = document.createElement("strong"); shift.textContent = `${pct(Number(combo.prior))} -> ${pct(Number(combo.posterior))}`;
      article.append(identity, track, shift); return article;
    });
    $("rangeInferenceCombos").replaceChildren(...rows);
    $("rangeInferenceProvenance").textContent = `${result.provenance.method}. ${result.provenance.coverage}. ${result.provenance.caveat}`;
    $("rangeInferenceResults").scrollIntoView({behavior:"smooth", block:"nearest"});
  } catch (error) {
    $("rangeInferenceError").textContent = error.message;
    $("rangeInferenceError").classList.remove("hidden");
  } finally {
    $("conditionRangeButton").disabled = false;
    $("conditionRangeButton").textContent = "Condition range";
  }
}

function practiceStreet(board) {
  return ({0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"})[board.length] || "Decision";
}

async function startSolverPractice() {
  $("solverPracticeError").classList.add("hidden");
  $("practiceRangeNodeButton").disabled = true;
  try {
    const challenge = await api("/api/solver-practice/sessions", {
      method: "POST",
      body: JSON.stringify({public_fingerprint: $("rangeNodeSelect").value}),
    });
    state.solverPractice = challenge;
    $("solverPracticePanel").classList.remove("hidden");
    $("solverPracticeResult").classList.add("hidden");
    $("solverPracticeStreet").textContent = practiceStreet(challenge.board);
    $("solverPracticePosition").textContent = challenge.hero_position;
    $("solverPracticePot").textContent = `${Number(challenge.pot_bb).toFixed(1)}bb`;
    $("solverPracticeStack").textContent = `${Number(challenge.effective_stack_bb).toFixed(1)}bb`;
    $("solverPracticeNode").textContent = challenge.node_id;
    renderCards($("solverPracticeHand"), challenge.hero_cards, 2);
    renderCards($("solverPracticeBoard"), challenge.board, challenge.board.length ? 5 : 0);
    $("solverPracticeHistory").replaceChildren(...challenge.action_history.map((text) => {
      const item = document.createElement("li"); item.textContent = text; return item;
    }));
    const actions = challenge.actions.map((action) => {
      const button = document.createElement("button");
      button.className = "action-button solver-practice-action";
      button.textContent = action.label;
      button.addEventListener("click", () => submitSolverPractice(action.action_id));
      return button;
    });
    $("solverPracticeActions").replaceChildren(...actions);
    $("solverPracticePanel").scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    $("solverPracticeError").textContent = error.message;
    $("solverPracticeError").classList.remove("hidden");
    $("solverPracticePanel").classList.remove("hidden");
  } finally {
    $("practiceRangeNodeButton").disabled = false;
  }
}

async function submitSolverPractice(actionId) {
  document.querySelectorAll(".solver-practice-action").forEach((button) => { button.disabled = true; });
  $("solverPracticeError").classList.add("hidden");
  try {
    const result = await api(`/api/solver-practice/sessions/${state.solverPractice.session_id}/decisions`, {
      method: "POST",
      body: JSON.stringify({action_id: actionId}),
    });
    $("solverPracticeResult").classList.remove("hidden");
    $("solverPracticeGrade").className = `grade-badge ${result.grade}`;
    $("solverPracticeGrade").textContent = result.grade.replace("_", " ");
    $("solverPracticeResultTitle").textContent = result.grade === "major_leak" ? "Rebuild this decision" : "Decision reviewed";
    $("solverPracticeLoss").textContent = `${Number(result.ev_loss_bb).toFixed(2)}bb`;
    $("solverPracticeFrequency").textContent = pct(Number(result.chosen_frequency));
    $("solverPracticeChosenEv").textContent = `${Number(result.chosen_ev_bb).toFixed(2)}bb`;
    $("solverPracticeBestEv").textContent = `${Number(result.best_ev_bb).toFixed(2)}bb`;
    const rows = result.strategy.map((action) => {
      const row = document.createElement("div"); row.className = "strategy-row";
      const label = document.createElement("strong"); label.textContent = action.label;
      const track = document.createElement("div"); track.className = "bar-track";
      const fill = document.createElement("span"); fill.style.width = `${Number(action.frequency) * 100}%`; track.appendChild(fill);
      const frequency = document.createElement("small"); frequency.textContent = pct(Number(action.frequency));
      const ev = document.createElement("small"); ev.textContent = `${Number(action.ev_bb).toFixed(2)} EV`;
      row.append(label, track, frequency, ev); return row;
    });
    $("solverPracticeStrategy").replaceChildren(...rows);
    $("solverPracticeProvenance").textContent = `Imported solution · ${result.source} ${result.source_version} · ${result.node_id}`;
    $("solverPracticeResult").scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    $("solverPracticeError").textContent = error.message;
    $("solverPracticeError").classList.remove("hidden");
    document.querySelectorAll(".solver-practice-action").forEach((button) => { button.disabled = false; });
  }
}

async function createLiveSession(tableId = $("liveTableId").value.trim()) {
  if (!tableId) throw new Error("Table identity is required");
  const session = await api("/api/live/sessions", {
    method: "POST",
    body: JSON.stringify({table_id: tableId}),
  });
  state.liveSession = session;
  state.liveResult = null;
  $("liveTableId").value = session.table_id;
  $("liveSessionStatus").textContent = `Session ${session.session_id.slice(0, 10)} · awaiting revision 0`;
  $("liveResults").classList.add("hidden");
  return session;
}

async function ensureLiveSession(tableId) {
  if (!state.liveSession || state.liveSession.table_id !== tableId) {
    return createLiveSession(tableId);
  }
  return state.liveSession;
}

async function loadLiveSample() {
  $("liveError").classList.add("hidden");
  try {
    const sample = await api("/api/sample-live-state");
    $("liveStateInput").value = JSON.stringify(sample, null, 2);
    $("liveTableId").value = sample.table_id;
    $("liveSessionStatus").textContent = "Matched example loaded · analyze to start a feed session";
  } catch (error) {
    $("liveError").textContent = error.message;
    $("liveError").classList.remove("hidden");
  }
}

async function loadVisualObservation() {
  $("visualObservationError").classList.add("hidden");
  try {
    const observation = await api("/api/sample-visual-observation");
    $("visualObservationInput").value = JSON.stringify(observation, null, 2);
    $("liveTableId").value = observation.fields.table_id.value;
    $("visualObservationState").textContent = "Frame loaded";
    $("visualObservationDetail").textContent = `Frame ${observation.frame.frame_id.slice(0, 10)} · submit it for confidence and stability checks.`;
  } catch (error) {
    $("visualObservationError").textContent = error.message;
    $("visualObservationError").classList.remove("hidden");
  }
}

function renderVisualTransition(audit) {
  const list = $("visualInvariantList");
  const passedLabels = {
    table_identity: "Table identity matches the live session",
    normalized_action_tokens: "Action history uses normalized tokens",
    revision_forward: "Revision advances monotonically",
    board_prefix: "Board preserves the previous street prefix",
    action_history_prefix: "Action history preserves prior actions",
    hero_cards_immutable: "Hero cards remain unchanged",
    game_immutable: "Game configuration remains unchanged",
    hero_position_immutable: "Hero position remains unchanged",
    rake_model_immutable: "Rake model remains unchanged",
    utility_model_immutable: "Utility model remains unchanged",
    player_count_nonincreasing: "Player count does not increase",
    pot_nondecreasing: "Pot does not decrease",
    heads_up_effective_stack_nonincreasing: "Heads-up effective stack does not increase",
    unchanged_state_pot: "Pot is stable without public-state progress",
    unchanged_state_stack: "Effective stack is stable without public-state progress",
    unchanged_state_call_price: "Call price is stable without public-state progress",
  };
  list.innerHTML = "";
  list.classList.toggle("hidden", !audit);
  if (!audit) return;
  const rows = audit.status === "rejected"
    ? audit.violations
    : audit.checks.filter((check) => check.passed).slice(0, 5);
  for (const row of rows) {
    const item = document.createElement("li");
    item.className = row.passed ? "passed" : "failed";
    const detail = row.passed ? (passedLabels[row.code] || row.code) : row.message;
    item.textContent = `${row.passed ? "Passed" : "Blocked"}: ${detail} (${row.code})`;
    list.appendChild(item);
  }
  for (const warning of (audit.warnings || [])) {
    const item = document.createElement("li");
    item.className = "warning";
    item.textContent = `Warning: ${warning.message} (${warning.code})`;
    list.appendChild(item);
  }
}

async function submitVisualObservation() {
  $("visualObservationError").classList.add("hidden");
  $("submitVisualObservationButton").disabled = true;
  try {
    const observation = JSON.parse($("visualObservationInput").value);
    if (!observation || Array.isArray(observation) || typeof observation !== "object") throw new Error("Visual observation JSON must be an object");
    const tableId = String(observation.fields?.table_id?.value || "").trim();
    const session = await ensureLiveSession(tableId);
    const result = await api(`/api/live/sessions/${session.session_id}/visual-observations`, {
      method: "POST",
      body: JSON.stringify(observation),
    });
    $("visualObservationState").textContent = result.status.replaceAll("_", " ");
    $("visualObservationStatus").classList.toggle("invalid", result.status === "invalid_transition");
    if (result.status === "low_confidence") {
      $("visualObservationDetail").textContent = `Blocked fields: ${result.low_confidence_fields.join(", ")} · minimum ${pct(Number(result.minimum_confidence))}.`;
    } else if (result.status === "pending_stability") {
      $("visualObservationDetail").textContent = `${result.observed_stable_frames} of ${result.required_stable_frames} distinct consistent frames · load another frame.`;
    } else if (result.status === "invalid_transition") {
      const violations = result.transition?.violations || [];
      $("visualObservationDetail").textContent = `Blocked before strategy: ${violations.map((row) => row.message).join("; ")}.`;
    } else {
      $("visualObservationDetail").textContent = `Stable revision ${result.revision} · mean confidence ${pct(Number(result.mean_confidence))} · screenshot ${result.evidence.image_sha256.slice(0, 12)}.`;
    }
    renderVisualTransition(result.transition);
    if (result.payload) $("liveStateInput").value = JSON.stringify(result.payload, null, 2);
    if (result.analysis) {
      state.liveSession.last_revision = result.analysis.last_revision;
      $("liveSessionStatus").textContent = `Session ${session.session_id.slice(0, 10)} · ${result.analysis.state.hand_id} · revision ${result.analysis.last_revision}`;
      renderLiveResult(result.analysis);
    }
  } catch (error) {
    $("visualObservationError").textContent = error.message;
    $("visualObservationError").classList.remove("hidden");
  } finally {
    $("submitVisualObservationButton").disabled = false;
  }
}

async function analyzeLiveState() {
  $("liveError").classList.add("hidden");
  $("analyzeLiveStateButton").disabled = true;
  try {
    const payload = JSON.parse($("liveStateInput").value);
    if (!payload || Array.isArray(payload) || typeof payload !== "object") throw new Error("Live-state JSON must be an object");
    const session = await ensureLiveSession(String(payload.table_id || "").trim());
    const result = await api(`/api/live/sessions/${session.session_id}/states`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.liveResult = result;
    state.liveSession.last_revision = result.last_revision;
    $("liveSessionStatus").textContent = `Session ${session.session_id.slice(0, 10)} · ${result.state.hand_id} · revision ${result.last_revision}`;
    renderLiveResult(result);
  } catch (error) {
    $("liveError").textContent = error.message;
    $("liveError").classList.remove("hidden");
  } finally {
    $("analyzeLiveStateButton").disabled = false;
  }
}

async function pollLiveHistory({silent = false} = {}) {
  if (livePollInProgress) return;
  const path = $("liveHistoryPath").value.trim();
  if (!path) {
    if (!silent) {
      $("liveError").textContent = "A growing hand-history file or folder is required";
      $("liveError").classList.remove("hidden");
    }
    return;
  }
  livePollInProgress = true;
  $("liveError").classList.add("hidden");
  $("pollLiveHistoryButton").disabled = true;
  try {
    localStorage.setItem("pokerCoach.liveHistoryPath", path);
    const session = await ensureLiveSession($("liveTableId").value.trim());
    const poll = await api(`/api/live/sessions/${session.session_id}/capture/polls`, {
      method: "POST",
      body: JSON.stringify({path}),
    });
    $("liveCaptureStatus").className = poll.status === "state_ready" ? "ready" : "waiting";
    $("liveCaptureStatus").textContent = `${poll.status.replaceAll("_", " ")} · ${poll.message}`;
    if (poll.payload) $("liveStateInput").value = JSON.stringify(poll.payload, null, 2);
    if (poll.analysis) {
      const changedAnalysis = !state.liveResult || state.liveResult.state_id !== poll.analysis.state_id;
      state.liveSession.last_revision = poll.analysis.last_revision;
      $("liveSessionStatus").textContent = `Session ${session.session_id.slice(0, 10)} · ${poll.analysis.state.hand_id} · revision ${poll.analysis.last_revision}`;
      if (changedAnalysis) renderLiveResult(poll.analysis);
    }
  } catch (error) {
    $("liveCaptureStatus").className = "waiting";
    $("liveCaptureStatus").textContent = "Capture poll failed";
    if (!silent) {
      $("liveError").textContent = error.message;
      $("liveError").classList.remove("hidden");
    }
  } finally {
    livePollInProgress = false;
    $("pollLiveHistoryButton").disabled = false;
  }
}

function configureLivePolling() {
  if (livePollTimer) clearInterval(livePollTimer);
  livePollTimer = null;
  localStorage.setItem("pokerCoach.autoPollLiveHistory", $("autoPollLiveHistory").checked ? "1" : "0");
  if ($("autoPollLiveHistory").checked) {
    pollLiveHistory();
    livePollTimer = setInterval(() => pollLiveHistory({silent: true}), 1000);
  }
}

function renderLiveResult(result) {
  state.liveResult = result;
  const tableState = result.state;
  const match = result.match;
  $("liveResults").classList.remove("hidden");
  $("liveDecisionFeedback").classList.add("hidden");
  $("liveStreet").textContent = tableState.street;
  $("livePosition").textContent = tableState.hero_position;
  $("livePot").textContent = `${Number(tableState.pot_bb).toFixed(1)}bb`;
  $("liveStack").textContent = `${Number(tableState.effective_stack_bb).toFixed(1)}bb`;
  $("liveSpr").textContent = Number(result.math.spr).toFixed(2);
  $("liveCallThreshold").textContent = Number(tableState.to_call_bb) > 0 ? pct(Number(result.math.call_break_even_equity)) : "No call";
  renderCards($("liveHeroCards"), tableState.hero_cards, 2);
  renderCards($("liveBoardCards"), tableState.board, tableState.board.length ? 5 : 0);
  const texture = result.texture;
  const textureLabels = [
    texture.pairing,
    texture.suit_texture,
    texture.straight_texture,
    texture.hero.made_hand,
    texture.hero.flush_draw !== "none" ? texture.hero.flush_draw : null,
    texture.hero.straight_draw !== "none" ? texture.hero.straight_draw : null,
  ].filter(Boolean);
  $("liveTextureTags").replaceChildren(...textureLabels.map((label) => {
    const tag = document.createElement("span");
    tag.textContent = label.replaceAll("_", " ");
    return tag;
  }));
  $("liveTextureFacts").replaceChildren(...texture.facts.map((text) => {
    const item = document.createElement("li"); item.textContent = text; return item;
  }));
  $("liveTextureCaveat").textContent = texture.range_caveat;
  $("liveActionHistory").replaceChildren(...tableState.action_history.map((text) => {
    const item = document.createElement("li"); item.textContent = text; return item;
  }));
  $("liveWarnings").replaceChildren(...result.warnings.map((text) => {
    const warning = document.createElement("div"); warning.className = "live-warning"; warning.textContent = text; return warning;
  }));
  if (!match) {
    $("liveResultTitle").textContent = "No covered solver node";
    $("liveMatchStatus").className = "grade-badge major_leak";
    $("liveMatchStatus").textContent = "unmatched";
    $("liveStrategyPanel").classList.add("hidden");
    $("liveUnmatched").classList.remove("hidden");
    $("liveUnmatched").textContent = "Import a solution for this exact game, player count, position, stack, pot, action history, board, and private hand before using a recommendation.";
  } else {
    $("liveResultTitle").textContent = "Solver node matched";
    $("liveMatchStatus").className = `grade-badge ${match.confidence === "exact" ? "" : "review"}`;
    $("liveMatchStatus").textContent = `${match.confidence} · ${pct(Number(match.score))}`;
    $("liveStrategyPanel").classList.remove("hidden");
    $("liveUnmatched").classList.add("hidden");
    $("liveNodeId").textContent = match.node_id;
    $("liveDominantAction").textContent = `Dominant mix: ${match.dominant_action || "none"} · Max EV: ${match.max_ev_action || "none"}`;
    const bars = match.actions.map((action) => {
      const row = document.createElement("div"); row.className = "strategy-row";
      const label = document.createElement("strong"); label.textContent = action.label;
      const track = document.createElement("div"); track.className = "bar-track";
      const fill = document.createElement("span"); fill.style.width = `${Number(action.frequency) * 100}%`; track.appendChild(fill);
      const frequency = document.createElement("small"); frequency.textContent = pct(Number(action.frequency));
      const ev = document.createElement("small"); ev.textContent = `${Number(action.ev_bb).toFixed(2)} EV`;
      row.append(label, track, frequency, ev); return row;
    });
    $("liveStrategyBars").replaceChildren(...bars);
    const actions = match.actions.map((action) => {
      const button = document.createElement("button");
      button.className = "action-button live-decision-action";
      button.textContent = `Record ${action.label}`;
      button.addEventListener("click", () => recordLiveDecision(action.action));
      return button;
    });
    $("liveDecisionButtons").replaceChildren(...actions);
    $("liveProvenance").textContent = `${match.source} ${match.source_version} · ${match.card_match} cards · state ${result.state_id.slice(0, 16)}`;
  }
  $("liveResults").scrollIntoView({behavior: "smooth", block: "nearest"});
}

async function recordLiveDecision(actionId) {
  document.querySelectorAll(".live-decision-action").forEach((button) => { button.disabled = true; });
  $("liveError").classList.add("hidden");
  try {
    const result = await api(`/api/live/sessions/${state.liveSession.session_id}/decisions`, {
      method: "POST",
      body: JSON.stringify({revision: state.liveResult.last_revision, action_id: actionId}),
    });
    $("liveDecisionFeedback").classList.remove("hidden");
    $("liveDecisionGrade").className = `grade-badge ${result.grade}`;
    $("liveDecisionGrade").textContent = result.grade.replace("_", " ");
    $("liveDecisionLoss").textContent = `${Number(result.ev_loss_bb).toFixed(2)}bb`;
    $("liveDecisionMix").textContent = pct(Number(result.chosen_frequency));
    $("liveDecisionEv").textContent = `${Number(result.chosen_ev_bb).toFixed(2)}bb`;
    $("liveDecisionBestEv").textContent = `${Number(result.best_ev_bb).toFixed(2)}bb`;
    $("liveDecisionFeedback").scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    $("liveError").textContent = error.message;
    $("liveError").classList.remove("hidden");
    document.querySelectorAll(".live-decision-action").forEach((button) => { button.disabled = false; });
  }
}

async function loadRangeStrategies() {
  try { renderRangeExplorer(await api("/api/range-strategies")); } catch (_) { /* optional until solutions exist */ }
}

function updateSolverExportFormat() {
  const csv = $("solverExportFormat").value === "tabular-csv-v1";
  $("solverExportLabel").textContent = csv ? "One action per CSV row" : "Versioned solver-bundle JSON";
  $("solverBundleInput").placeholder = csv
    ? "schema_version,source,source_version,node_id,game,…"
    : "{ \"schema_version\": \"1.0.0\", … }";
  $("solverImportResult").classList.add("hidden");
}

function showFatal(error) {
  $("healthLabel").textContent = error.message;
  document.querySelector(".health").classList.remove("ready");
}

async function analyzeHands() {
  $("reviewError").classList.add("hidden");
  try {
    const report = await api("/api/analyze-hand-history", { method: "POST", body: JSON.stringify({ hand_history: $("handHistory").value }) });
    renderReport(report);
  } catch (error) {
    $("reviewError").textContent = error.message;
    $("reviewError").classList.remove("hidden");
  }
}

function renderScanStatus(scan) {
  $("scanStatus").classList.remove("hidden");
  $("scanLibraryHands").textContent = `${scan.database_hands} hand${scan.database_hands === 1 ? "" : "s"}`;
  $("scanImported").textContent = `${scan.inserted} new · ${scan.updated} updated`;
  $("scanFiles").textContent = `${scan.files_seen} seen · ${scan.skipped_files} unchanged`;
  $("scanPending").textContent = String(scan.incomplete_blocks);
  const parts = [];
  if (scan.unstable_files) parts.push(`${scan.unstable_files} changing file(s) will be retried`);
  if (scan.errors.length) parts.push(`${scan.errors.length} parse/read error(s)`);
  if (!parts.length) parts.push("Scan completed cleanly");
  $("scanMessage").textContent = `${parts.join(" · ")} · ${scan.folder}`;
  $("scanStatus").classList.toggle("warning", scan.errors.length > 0);
}

async function scanFolder({quiet = false} = {}) {
  if (scanInProgress) return;
  const folder = $("handHistoryFolder").value.trim();
  if (!folder) {
    if (!quiet) {
      $("scanError").textContent = "Enter the folder where PokerStars saves English hand histories.";
      $("scanError").classList.remove("hidden");
    }
    return;
  }
  scanInProgress = true;
  $("scanError").classList.add("hidden");
  $("scanFolderButton").disabled = true;
  $("scanFolderButton").textContent = "Scanning…";
  try {
    localStorage.setItem("pokerCoach.handHistoryFolder", folder);
    localStorage.setItem("pokerCoach.recursiveScan", $("recursiveScan").checked ? "1" : "0");
    const report = await api("/api/scan-hand-history-folder", {
      method: "POST",
      body: JSON.stringify({folder, recursive: $("recursiveScan").checked}),
    });
    renderScanStatus(report.scan);
    renderReport(report);
  } catch (error) {
    $("scanError").textContent = error.message;
    $("scanError").classList.remove("hidden");
  } finally {
    scanInProgress = false;
    $("scanFolderButton").disabled = false;
    $("scanFolderButton").textContent = "Scan now";
  }
}

async function loadLibrary() {
  $("scanError").classList.add("hidden");
  try {
    const report = await api("/api/hand-history-library");
    renderReport(report);
  } catch (error) {
    $("scanError").textContent = error.message;
    $("scanError").classList.remove("hidden");
  }
}

function configureAutoScan() {
  if (scanTimer !== null) window.clearInterval(scanTimer);
  scanTimer = null;
  localStorage.setItem("pokerCoach.autoScan", $("autoScan").checked ? "1" : "0");
  if ($("autoScan").checked) {
    scanFolder({quiet: true});
    scanTimer = window.setInterval(() => scanFolder({quiet: true}), 5000);
  }
}

function renderReport(report) {
  $("reviewResults").classList.remove("hidden");
  $("resultHands").textContent = report.hands;
  $("resultPlayers").textContent = Object.keys(report.player_profiles).length;
  $("resultMatches").textContent = report.solution_review.matched_decisions;
  let rangeUpdateMetric = $("resultRangeUpdates");
  if (!rangeUpdateMetric) {
    const wrapper = document.createElement("div");
    const label = document.createElement("span"); label.textContent = "Range updates";
    rangeUpdateMetric = document.createElement("strong"); rangeUpdateMetric.id = "resultRangeUpdates";
    wrapper.append(label, rangeUpdateMetric); document.querySelector(".result-summary").appendChild(wrapper);
  }
  rangeUpdateMetric.textContent = report.opponent_range_review.conditioned_actions;
  $("resultEvLoss").textContent = `${Number(report.solution_review.leak_summary.total_ev_loss_bb).toFixed(2)}bb`;
  $("resultCoverage").textContent = pct(Number(report.solution_review.leak_summary.coverage));
  const errorCount = report.hand_reports.filter((hand) => Number(hand.reconciliation_error) !== 0).length;
  $("resultErrors").textContent = errorCount;
  const rows = report.hand_reports.map((hand) => {
    const item = document.createElement("article"); item.className = "report-item";
    const text = document.createElement("div");
    const title = document.createElement("strong"); title.textContent = `Hand #${hand.hand_id} · ${hand.table}`;
    const summary = document.createElement("span"); summary.textContent = `${hand.players} players · ${hand.actions} actions · ${hand.hero_decisions.length} hero decisions`;
    text.append(title, summary);
    const audit = document.createElement("strong"); audit.className = Number(hand.reconciliation_error) === 0 ? "ok" : ""; audit.textContent = Number(hand.reconciliation_error) === 0 ? "Pot reconciled" : `Pot error ${hand.reconciliation_error}`;
    item.append(text, audit); return item;
  });
  $("handReports").replaceChildren(...rows);
  renderOpponentRangeTimelines(report.opponent_range_review);
  renderRangeCalibration(report.range_calibration);
  renderDrillQueue(report.study_queue);
  const profileCards = Object.entries(report.profile_summaries).map(([player, summary]) => {
    const card = document.createElement("article"); card.className = "profile-card";
    const head = document.createElement("div"); head.className = "profile-head";
    const identity = document.createElement("div");
    const name = document.createElement("strong"); name.textContent = player;
    const label = document.createElement("span"); label.textContent = summary.style_label;
    identity.append(name, label);
    const confidence = document.createElement("small"); confidence.textContent = `${summary.confidence} confidence`;
    head.append(identity, confidence);
    const metrics = document.createElement("div"); metrics.className = "profile-metrics";
    [["VPIP", "vpip"], ["PFR", "pfr"], ["Aggression", "aggressive_action"]].forEach(([display, key]) => {
      const metric = summary.metrics[key];
      const row = document.createElement("div");
      const metricLabel = document.createElement("span"); metricLabel.textContent = display;
      const value = document.createElement("strong"); value.textContent = pct(Number(metric.posterior_mean));
      const sample = document.createElement("small"); sample.textContent = `${Number(metric.opportunities).toFixed(0)} opportunities`;
      row.append(metricLabel, value, sample); metrics.appendChild(row);
    });
    const caveat = document.createElement("p");
    caveat.textContent = summary.confidence === "limited" ? "Sample is too small for a stable style label; estimates remain strongly uncertainty-weighted." : "Style label is derived from opportunity-based posterior estimates.";
    const insights = (report.exploit_insights[player] || []).slice(0, 3);
    const insightList = document.createElement("div"); insightList.className = "exploit-list";
    insights.forEach((insight) => {
      const row = document.createElement("div"); row.className = "exploit-insight";
      const label = document.createElement("strong"); label.textContent = insight.title;
      const evidence = document.createElement("small"); evidence.textContent = `${insight.confidence} · ${Number(insight.opportunities).toFixed(0)} opportunities · ${pct(Number(insight.posterior_mean))}`;
      const adjustment = document.createElement("span");
      adjustment.textContent = insight.actionability === "observe_only" ? `Watch: ${insight.rationale}` : insight.adjustment;
      row.append(label, evidence, adjustment); insightList.appendChild(row);
    });
    card.append(head, metrics, caveat, insightList); return card;
  });
  $("playerProfiles").replaceChildren(...profileCards);
}

function renderOpponentRangeTimelines(review) {
  const timelines = review.timelines.map((timeline) => {
    const article = document.createElement("article"); article.className = "range-timeline-card";
    const head = document.createElement("div"); head.className = "range-timeline-head";
    const identity = document.createElement("div");
    const title = document.createElement("strong"); title.textContent = `${timeline.opponent} range`;
    const state = document.createElement("span"); state.textContent = `Hand #${timeline.hand_id} | ${timeline.position} | ${timeline.events.length} compatible actions`;
    identity.append(title, state);
    const coverage = document.createElement("small"); coverage.textContent = `${pct(Number(review.coverage))} public-state coverage`;
    head.append(identity, coverage);
    const eventList = document.createElement("div"); eventList.className = "range-timeline-events";
    timeline.events.forEach((event, index) => {
      const row = document.createElement("section"); row.className = `range-timeline-event ${event.status}`;
      const marker = document.createElement("div"); marker.className = "range-timeline-marker"; marker.textContent = String(index + 1).padStart(2, "0");
      const copy = document.createElement("div"); copy.className = "range-timeline-copy";
      const eventHead = document.createElement("div"); eventHead.className = "range-event-head";
      const action = document.createElement("strong"); action.textContent = `${event.street} ${event.observed_action || "uncovered action"}`;
      const confidence = document.createElement("span"); confidence.textContent = `${event.match.confidence} public match | ${event.pot_bb}bb pot`;
      eventHead.append(action, confidence); copy.appendChild(eventHead);
      if (event.status === "conditioned") {
        const posterior = event.posterior;
        const transition = document.createElement("p");
        const skipped = Number(event.prior_transition.unmatched_actions_skipped || 0);
        transition.textContent = event.prior_transition.mode === "posterior_carried"
          ? `Prior carried from action ${event.prior_transition.from_action_index}${skipped ? ` across ${skipped} unmatched action${skipped === 1 ? "" : "s"}` : ""}.`
          : `Uniform prior reset: ${event.prior_transition.reason.replaceAll("_", " ")}.`;
        copy.appendChild(transition);
        const metrics = document.createElement("div"); metrics.className = "range-event-metrics";
        const evidence = document.createElement("span"); evidence.textContent = `Evidence ${pct(Number(posterior.action_probability_under_prior))}`;
        const entropy = document.createElement("span"); entropy.textContent = `Entropy ${Number(posterior.information.prior_entropy_bits).toFixed(3)} -> ${Number(posterior.information.posterior_entropy_bits).toFixed(3)} bits`;
        metrics.append(evidence, entropy); copy.appendChild(metrics);
        const combos = document.createElement("div"); combos.className = "range-event-combos";
        posterior.combos.slice(0, 4).forEach((combo) => {
          const comboRow = document.createElement("div");
          const comboName = document.createElement("span"); comboName.textContent = `${combo.combo} (${combo.hand_class})`;
          const bar = document.createElement("i"); bar.style.setProperty("--posterior", `${Math.max(0, Math.min(100, Number(combo.posterior) * 100))}%`);
          const weight = document.createElement("strong"); weight.textContent = `${pct(Number(combo.prior))} -> ${pct(Number(combo.posterior))}`;
          comboRow.append(comboName, bar, weight); combos.appendChild(comboRow);
        });
        copy.appendChild(combos);
      } else {
        const error = document.createElement("p"); error.className = "range-event-error"; error.textContent = event.conditioning_error;
        copy.appendChild(error);
      }
      row.append(marker, copy); eventList.appendChild(row);
    });
    const caveat = document.createElement("p"); caveat.className = "range-timeline-caveat"; caveat.textContent = review.caveat;
    article.append(head, eventList, caveat); return article;
  });
  if (!timelines.length) {
    const empty = document.createElement("p"); empty.className = "empty-state";
    empty.textContent = "Import compatible exact-combo solver nodes to reconstruct opponent range updates from saved actions.";
    $("opponentRangeTimelines").replaceChildren(empty);
  } else {
    $("opponentRangeTimelines").replaceChildren(...timelines);
  }
}

function renderRangeCalibration(calibration) {
  const aggregate = calibration.aggregate;
  if (!aggregate.scored_predictions) {
    const empty = document.createElement("p"); empty.className = "empty-state";
    empty.textContent = `${aggregate.unrevealed_timelines} compatible timeline${aggregate.unrevealed_timelines === 1 ? "" : "s"} had no revealed opponent cards. Calibration begins only when a saved showdown exposes a supported exact combo.`;
    $("rangeCalibration").replaceChildren(empty); return;
  }
  const panel = document.createElement("article"); panel.className = "calibration-card";
  const metrics = document.createElement("div"); metrics.className = "calibration-metrics";
  const metricRows = [
    ["Scored", String(aggregate.scored_predictions)],
    ["Support", pct(Number(aggregate.support_coverage))],
    ["Log loss", aggregate.mean_log_loss_bits === "infinite" ? "infinite" : `${Number(aggregate.mean_log_loss_bits).toFixed(3)} bits`],
    ["Brier", Number(aggregate.mean_multiclass_brier_score).toFixed(3)],
    ["Top combo", pct(Number(aggregate.top_1_accuracy))],
    ["Cal. error", Number(aggregate.expected_calibration_error).toFixed(3)],
  ];
  metricRows.forEach(([labelText, valueText]) => {
    const row = document.createElement("div");
    const label = document.createElement("span"); label.textContent = labelText;
    const value = document.createElement("strong"); value.textContent = valueText;
    row.append(label, value); metrics.appendChild(row);
  });
  const predictions = document.createElement("div"); predictions.className = "calibration-predictions";
  calibration.timelines.filter((timeline) => timeline.status === "scored").forEach((timeline) => {
    const group = document.createElement("section");
    const head = document.createElement("div"); head.className = "calibration-head";
    const identity = document.createElement("strong"); identity.textContent = `${timeline.opponent} | ${timeline.revealed_cards.join(" ")}`;
    const hand = document.createElement("span"); hand.textContent = `Hand #${timeline.hand_id}`;
    head.append(identity, hand); group.appendChild(head);
    timeline.predictions.filter((prediction) => prediction.scored).forEach((prediction) => {
      const row = document.createElement("div"); row.className = "calibration-prediction";
      const state = document.createElement("div");
      const action = document.createElement("strong"); action.textContent = `${prediction.street} ${prediction.observed_action}`;
      const combo = document.createElement("span"); combo.textContent = `${prediction.actual_combo} | rank ${prediction.posterior_rank}/${prediction.support_combos}`;
      state.append(action, combo);
      const trajectory = document.createElement("div"); trajectory.className = "calibration-trajectory";
      const probability = document.createElement("strong"); probability.textContent = `${pct(Number(prediction.actual_prior_probability))} -> ${pct(Number(prediction.actual_posterior_probability))}`;
      const loss = document.createElement("span"); loss.textContent = prediction.log_loss_bits === "infinite" ? "infinite log loss" : `${Number(prediction.log_loss_bits).toFixed(3)} bits log loss`;
      trajectory.append(probability, loss); row.append(state, trajectory); group.appendChild(row);
    });
    predictions.appendChild(group);
  });
  const buckets = document.createElement("div"); buckets.className = "calibration-buckets";
  calibration.calibration_buckets.forEach((bucket) => {
    const row = document.createElement("div");
    const label = document.createElement("span"); label.textContent = `${Math.round(Number(bucket.lower) * 100)}-${Math.round(Number(bucket.upper) * 100)}% | n=${bucket.observations}`;
    const track = document.createElement("i");
    const forecast = document.createElement("b"); forecast.style.setProperty("--forecast", `${Number(bucket.mean_forecast) * 100}%`); forecast.title = `Mean forecast ${pct(Number(bucket.mean_forecast))}`;
    const actual = document.createElement("em"); actual.style.setProperty("--actual", `${Number(bucket.empirical_hit_rate) * 100}%`); actual.title = `Observed ${pct(Number(bucket.empirical_hit_rate))}`;
    track.append(forecast, actual); row.append(label, track); buckets.appendChild(row);
  });
  const legend = document.createElement("p"); legend.className = "calibration-caveat"; legend.textContent = `${calibration.caveat} Bucket bars compare mean forecast (green) with observed rate (gold).`;
  panel.append(metrics, predictions, buckets, legend);
  $("rangeCalibration").replaceChildren(panel);
}

function renderDrillQueue(drillRows) {
  const drills = drillRows.map((drill, index) => {
    const article = document.createElement("article"); article.className = "drill-card";
    const rank = document.createElement("span"); rank.className = "drill-rank"; rank.textContent = String(index + 1).padStart(2, "0");
    const copy = document.createElement("div");
    const title = document.createElement("strong"); title.textContent = drill.title;
    const state = document.createElement("span"); state.textContent = `${drill.street} · ${drill.hero_position} · ${drill.hero_cards.join(" ")} · ${drill.board.join(" ")}`;
    const correction = document.createElement("span");
    correction.textContent = `Observed ${drill.observed_action}; highest-EV action ${drill.best_actions[0].action} (${pct(Number(drill.best_actions[0].frequency))})`;
    copy.append(title, state, correction);
    const loss = document.createElement("div"); loss.className = "drill-loss";
    const value = document.createElement("strong"); value.textContent = `${Number(drill.ev_loss_bb).toFixed(2)}bb`;
    const label = document.createElement("small"); label.textContent = `${pct(Number(drill.study.mastery))} mastery · ${drill.study.attempts} attempts`;
    loss.append(value, label);
    const ratings = document.createElement("div"); ratings.className = "rating-controls";
    ["again", "hard", "good", "easy"].forEach((rating) => {
      const button = document.createElement("button");
      button.textContent = rating;
      button.dataset.rating = rating;
      button.setAttribute("aria-label", `${rating} · ${drill.title}`);
      button.addEventListener("click", () => rateDrill(drill.drill_id, rating, button));
      ratings.appendChild(button);
    });
    copy.appendChild(ratings);
    article.append(rank, copy, loss); return article;
  });
  if (!drills.length) {
    const empty = document.createElement("p"); empty.className = "empty-state";
    empty.textContent = "Import matching solver nodes and analyze saved hands to generate drills.";
    $("drillQueue").replaceChildren(empty);
  } else {
    $("drillQueue").replaceChildren(...drills);
  }
}

async function rateDrill(drillId, rating, button) {
  const cardButtons = button.closest(".rating-controls").querySelectorAll("button");
  cardButtons.forEach((item) => { item.disabled = true; });
  try {
    await api(`/api/drills/${drillId}/attempts`, {method:"POST", body:JSON.stringify({rating})});
    const queue = await api("/api/drills");
    renderDrillQueue(queue.drills);
  } catch (error) {
    cardButtons.forEach((item) => { item.disabled = false; });
    $("reviewError").textContent = error.message;
    $("reviewError").classList.remove("hidden");
  }
}

document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".nav-button").forEach((item) => item.classList.toggle("active", item === button));
  $("trainerView").classList.toggle("active", button.dataset.view === "trainer");
  $("liveView").classList.toggle("active", button.dataset.view === "live");
  $("solverView").classList.toggle("active", button.dataset.view === "solver");
  $("reviewView").classList.toggle("active", button.dataset.view === "review");
}));

$("analyzeButton").addEventListener("click", analyzeHands);
$("scanFolderButton").addEventListener("click", () => scanFolder());
$("loadLibraryButton").addEventListener("click", loadLibrary);
$("autoScan").addEventListener("change", configureAutoScan);
$("solveRiverButton").addEventListener("click", solveRiver);
$("analyzeRangeMatchupButton").addEventListener("click", analyzeRangeMatchup);
$("loadSolverBundleButton").addEventListener("click", loadSolverBundle);
$("importSolverBundleButton").addEventListener("click", importSolverBundle);
$("solverExportFormat").addEventListener("change", updateSolverExportFormat);
$("rangeNodeSelect").addEventListener("change", renderSelectedRangeGroup);
$("rangeInferenceAction").addEventListener("change", () => {
  state.rangeInference = null;
  $("rangeInferenceResults").classList.add("hidden");
});
$("conditionRangeButton").addEventListener("click", conditionSelectedRange);
$("practiceRangeNodeButton").addEventListener("click", startSolverPractice);
$("createLiveSessionButton").addEventListener("click", async () => {
  $("liveError").classList.add("hidden");
  try { await createLiveSession(); } catch (error) { $("liveError").textContent = error.message; $("liveError").classList.remove("hidden"); }
});
$("loadLiveSampleButton").addEventListener("click", loadLiveSample);
$("analyzeLiveStateButton").addEventListener("click", analyzeLiveState);
$("loadVisualObservationButton").addEventListener("click", loadVisualObservation);
$("submitVisualObservationButton").addEventListener("click", submitVisualObservation);
$("pollLiveHistoryButton").addEventListener("click", () => pollLiveHistory());
$("autoPollLiveHistory").addEventListener("change", configureLivePolling);
document.querySelectorAll("[data-pot-fraction]").forEach((button) => button.addEventListener("click", () => {
  $("solverBet").value = (Number($("solverPot").value) * Number(button.dataset.potFraction)).toFixed(1);
}));
$("loadSampleButton").addEventListener("click", async () => {
  try {
    const sample = await api("/api/sample-hand");
    $("handHistory").value = sample.hand_history;
  } catch (error) {
    $("reviewError").textContent = error.message;
    $("reviewError").classList.remove("hidden");
  }
});

(async function init() {
  try {
    $("handHistoryFolder").value = localStorage.getItem("pokerCoach.handHistoryFolder") || "";
    $("recursiveScan").checked = localStorage.getItem("pokerCoach.recursiveScan") === "1";
    $("autoScan").checked = localStorage.getItem("pokerCoach.autoScan") === "1";
    $("liveHistoryPath").value = localStorage.getItem("pokerCoach.liveHistoryPath") || "";
    $("autoPollLiveHistory").checked = localStorage.getItem("pokerCoach.autoPollLiveHistory") === "1";
    const health = await api("/api/health");
    document.querySelector(".health").classList.add("ready");
    $("healthLabel").textContent = `${health.scenario_count} spots ready · local`;
    await startSession();
    await loadRangeStrategies();
    if (health.persistent_hand_library && health.database_hands > 0) await loadLibrary();
    if ($("autoScan").checked) configureAutoScan();
    if ($("autoPollLiveHistory").checked) configureLivePolling();
  } catch (error) { showFatal(error); }
})();
