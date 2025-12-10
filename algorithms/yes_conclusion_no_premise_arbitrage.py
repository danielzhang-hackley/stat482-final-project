"""
Calculate arbitrage opportunities that arise from the "buy yes conclusion, no premise" strategy
"""

import itertools
from typing import Set, DefaultDict, TypeVar

T = TypeVar('T')

def build_arbitrage_structures(df, implied_by, contradictions):
    """
    Convert Polymarket dataframe + implication graph + contradiction graph
    into the `events` and `mutual_exclusive` structures expected by find_arbitrage().

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a "price" column.
        Indices of df are event IDs (used in implied_by and contradictions).

    implied_by : dict[int, list[int]]
        Maps event index X → list of event indices A that imply X.

    contradictions : dict[int, list[int]]
        Maps event index E → list of event indices that contradict E.

    Returns
    -------
    (events, mutual_exclusive)
        events : dict
            event_id → {
                "price": float,
                "premises": [(premise_event_id, price)],
                # no max_simultaneous here; solver infers this automatically
            }

        mutual_exclusive : dict
            event_id → list of event_ids that are mutually exclusive with it.
            Ensures symmetry and no duplicates.
    """

    events = {}
    mutual_exclusive = {}

    # --- Build the events structure ---
    for event_id in df.index:
        price = float(df.loc[event_id, "price"])

        premises = []
        if event_id in implied_by:
            for prem in implied_by[event_id]:
                prem_price = float(df.loc[prem, "price"])
                premises.append((prem, prem_price))

        events[event_id] = {
            "price": price,
            "premises": premises
        }

    # --- Build the mutual_exclusive structure ---
    # We make contradictions symmetric: if A contradicts B, B contradicts A.
    for event_id in df.index:
        mutual_exclusive[event_id] = []

    for a, lst in contradictions.items():
        for b in lst:
            mutual_exclusive[a].append(b)
            mutual_exclusive[b].append(a)

    # Remove duplicates
    for e in mutual_exclusive:
        mutual_exclusive[e] = list(sorted(set(mutual_exclusive[e])))

    return events, mutual_exclusive


def calculate_m_optimal(conclusion, premise_names, mutual_exclusive: DefaultDict[T, Set[T]]):
    """
    Computes m = max number of premises that can occur together AND are compatible with X.

    conclusion: the name of event X
    premise_names: list of premises A_i implying X
    mutual_exclusive: dict mapping event -> list of mutually exclusive events
    """

    # Step 1: Filter out premises that are mutually exclusive with X
    compatible = []
    for A in premise_names:
        if conclusion not in mutual_exclusive[A]:
            compatible.append(A)

    if not compatible:
        return 0

    # Build exclusivity lookup inside the compatible set
    excl = {a: mutual_exclusive[a] for a in compatible}

    # Step 2: Compute maximum independent set on the compatible nodes
    for r in range(len(compatible) + 1, 0, -1):
        for subset in itertools.combinations(compatible, r):
            for a, b in itertools.combinations(subset, 2):
                if b in excl[a] or a in excl[b]:
                    return r
    return 1


def find_arbitrage(events, mutual_exclusive):
    """
    Detects arbitrage opportunities of the form:
        Buy 1 No on each A_i and buy 1 Yes on X.

    events:
        dict: event_name -> {
            "price": float,
            "premises": [(premise_name, price)],
        }

    mutual_exclusive:
        dict: event_name -> [list of mutually exclusive events]
    """

    arb = {}

    for conclusion, data in events.items():
        pX = data["price"]
        premises = data.get("premises", [])
        premise_names = [name for name, _ in premises]
        premise_sum = sum(p for _, p in premises)

        # Oonly count premises that are compatible with X
        m = calculate_m_optimal(
            conclusion,
            premise_names,
            mutual_exclusive
        )

        # Arbitrage condition: sum p_i >= p_X + (m - 1)
        threshold = pX + (m - 1)

        if premise_sum >= threshold:
            arb[conclusion] = {
                "premise_sum": premise_sum,
                "threshold": threshold,
                "m": m,
                "premises": premises,
                "compatible_premises": [
                    A for A in premise_names
                    if conclusion not in mutual_exclusive.get(A, [])
                ],
                "price_X": pX,
                "guaranteed_profit": premise_sum - threshold
            }

    return arb
