# Sokoban Solver

A Sokoban solver for make.fun's "Build the best Sokoban solver" bounty.

## Approach

Every level from this competition's generator is one hallway plus 1–4
rooms (6x6 slots, 3–4 boxes each). A box pushed out of a room's door can
never come back, so a level's push-optimum P* is the sum of its rooms'
optima. This solver exploits that structure:

1. **Decompose.** The hallway is the player's row. Removing its floor
   cells splits the board into rooms; each room has one door, and the
   hallway cell in front of the door is the room's entry (a box can start
   on the door cell, so the player must be able to push it in from the
   hallway).
2. **Solve each room push-optimally.** A* over push-states (sorted box
   positions + canonical player region), with a min-cost box→goal
   assignment heuristic (brute-force for ≤5 boxes, using precomputed BFS
   distances), static dead-square pruning (reverse pull-reachability from
   the goals) and 2x2-block deadlock pruning. A deterministic node cap
   bounds each room; a greedy best-first search is the fallback so an
   answer is always produced.
3. **Stitch.** BFS walks connect the rooms through the hallway; each
   room's pushes expand into player moves (lowercase walks, uppercase
   pushes).

Everything is deterministic: budgets are node counts, never the clock.

## Files

- `main.ts` — the whole solver (Node 22, TypeScript erasable syntax, stdlib only)
- `README.md` — this file
- `LICENSE` — MIT
