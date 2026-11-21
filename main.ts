// Sokoban solver: per-room push-optimal A* with min-assignment heuristic,
// static dead squares and 2x2 deadlock pruning, then stitched together
// with BFS walks through the hallway.
//
// Why per-room: every level is one hallway plus 1-4 rooms (6x6 slots, 3-4
// boxes each). A box pushed out of a room's door can never come back, so a
// level's push-optimum P* is the sum of its rooms' optima. Solving each
// room independently keeps the search tiny; a whole-level search over 16
// boxes would be hopeless. The player enters a room through its door; the
// hallway cell in front of the door is part of the room's walkable space
// (a box can start ON the door cell and must be pushed in from outside).
//
// Determinism: every budget is a node count, never the clock. All
// tie-breaking is fixed. No Date, no Math.random, no environment reads.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const ROOM_CAP = 300_000;
const WHOLE_CAP = 300_000;

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  boxes: number[];
  player: number;
}

interface Room {
  cells: number[];
  cellSet: Set<number>;
  boxes: number[];
  goals: number[];
  door: number;
  entry: number;
}

interface Push {
  from: number;
  to: number;
  dir: number;
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall: boolean[] = new Array(w * h).fill(false);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// BFS distances from `from` over walkable cells (ignoring boxes). -1 = unreachable.
function bfsDist(from: number, walkable: boolean[], w: number, h: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  dist[from] = 0;
  const q: number[] = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!walkable[n] || dist[n] !== -1) continue;
      dist[n] = dist[c]! + 1;
      q.push(n);
    }
  }
  return dist;
}

// Shortest walk path from `from` to `to` over walkable cells avoiding `blocked`
// (a Set of box cells). Returns the list of cells after `from`, or null.
function bfsPath(from: number, to: number, walkable: boolean[], blocked: Set<number>, w: number, h: number): number[] | null {
  if (from === to) return [];
  const prev = new Int32Array(w * h).fill(-1);
  prev[from] = from;
  const q: number[] = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!walkable[n] || blocked.has(n) || prev[n] !== -1) continue;
      prev[n] = c;
      if (n === to) {
        const path: number[] = [];
        let t = n;
        while (t !== from) {
          path.push(t);
          t = prev[t]!;
        }
        path.reverse();
        return path;
      }
      q.push(n);
    }
  }
  return null;
}

// Letters for a walk path: `from` is the cell before path[0].
function pathLetters(from: number, path: number[], w: number): string {
  let s = '';
  let cur = from;
  for (const cell of path) {
    const d = cell - cur;
    if (d === -w) s += 'u';
    else if (d === w) s += 'd';
    else if (d === -1) s += 'l';
    else s += 'r';
    cur = cell;
  }
  return s;
}

// ---- Room decomposition -------------------------------------------------
// The hallway is the player's row. Removing its floor cells splits the
// board into the rooms; each room has exactly one door cell adjacent to
// the hallway, and `entry` is the hallway cell in front of that door.
function findRooms(lv: Level): Room[] {
  const { w, h, wall } = lv;
  const hallRow = Math.floor(lv.player / w);
  const inRoom = (i: number): boolean => !wall[i] && Math.floor(i / w) !== hallRow;
  const seen = new Set<number>();
  const rooms: Room[] = [];
  for (let i = 0; i < w * h; i++) {
    if (!inRoom(i) || seen.has(i)) continue;
    const comp: number[] = [];
    const q: number[] = [i];
    seen.add(i);
    while (q.length > 0) {
      const c = q.pop()!;
      comp.push(c);
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (inRoom(n) && !seen.has(n)) {
          seen.add(n);
          q.push(n);
        }
      }
    }
    const cellSet = new Set(comp);
    const boxes = lv.boxes.filter((b) => cellSet.has(b));
    const goals = lv.goals.filter((g) => cellSet.has(g));
    const doors = comp.filter((c) => {
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (!wall[n] && Math.floor(n / w) === hallRow) return true;
      }
      return false;
    });
    if (doors.length === 0) continue;
    doors.sort((a, b) => a - b);
    const door = doors[0]!;
    // Entry: the hallway cell adjacent to the door.
    let entry = -1;
    const dcx = door % w;
    const dcy = Math.floor(door / w);
    for (let d = 0; d < 4; d++) {
      const nx = dcx + DX[d]!;
      const ny = dcy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!wall[n] && Math.floor(n / w) === hallRow) {
        entry = n;
        break;
      }
    }
    if (entry < 0) continue;
    rooms.push({ cells: comp, cellSet, boxes, goals, door, entry });
  }
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// ---- Push-space search context -------------------------------------------
interface SearchCtx {
  w: number;
  h: number;
  walkable: boolean[]; // cells the player may stand on (room cells + entry)
  boxCells: boolean[]; // cells a box may occupy (room cells only)
  goals: number[];
  goalSet: Set<number>;
  goalDist: Int32Array[]; // goalDist[g][cell]: BFS distance cell -> goals[g]
  live: boolean[]; // static live squares for a box (pull-reachability)
  useAssignment: boolean; // brute-force min assignment (n <= 5), else sum-of-min
}

function buildCtx(w: number, h: number, wall: boolean[], roomCells: number[], entry: number, goals: number[]): SearchCtx {
  const walkable: boolean[] = new Array(w * h).fill(false);
  const boxCells: boolean[] = new Array(w * h).fill(false);
  for (const c of roomCells) {
    walkable[c] = true;
    boxCells[c] = true;
  }
  if (entry >= 0) walkable[entry] = true;
  const goalSet = new Set(goals);
  const goalDist = goals.map((g) => bfsDist(g, walkable, w, h));
  // Static dead squares: reverse pull BFS from the goals. A box at cell a
  // can be pulled to neighbour b when b is walkable and the player can
  // stand at 2a - b. Cells never reached are squares a box can never leave
  // towards a goal: a box there (off-goal) is dead. Over-approximation is
  // safe: it only prunes truly dead positions.
  const live: boolean[] = new Array(w * h).fill(false);
  const q: number[] = [];
  for (const g of goals) {
    live[g] = true;
    q.push(g);
  }
  for (let qi = 0; qi < q.length; qi++) {
    const b = q[qi]!;
    const bx = b % w;
    const by = Math.floor(b / w);
    for (let d = 0; d < 4; d++) {
      const ax = bx + DX[d]!;
      const ay = by + DY[d]!;
      if (ax < 0 || ay < 0 || ax >= w || ay >= h) continue;
      const a = ay * w + ax;
      if (!boxCells[a] || live[a]) continue;
      // Player stands at 2a - b to pull a -> b.
      const sx = 2 * ax - bx;
      const sy = 2 * ay - by;
      if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
      const s = sy * w + sx;
      if (!walkable[s]) continue;
      live[a] = true;
      q.push(a);
    }
  }
  return { w, h, walkable, boxCells, goals, goalSet, goalDist, live, useAssignment: goals.length <= 5 };
}

// Min-cost assignment of boxes to goals (admissible: one push moves one box
// one cell, so the assignment sum lower-bounds the remaining pushes).
function heuristic(ctx: SearchCtx, boxes: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (ctx.useAssignment) {
    // Brute-force permutations (n <= 5).
    const perm: number[] = [];
    for (let i = 0; i < n; i++) perm.push(i);
    let best = Infinity;
    const rec = (k: number, acc: number): void => {
      if (acc >= best) return;
      if (k === n) {
        best = acc;
        return;
      }
      for (let i = k; i < n; i++) {
        const tmp = perm[k]!;
        perm[k] = perm[i]!;
        perm[i] = tmp;
        const d = ctx.goalDist[perm[k]!]![boxes[k]!]!;
        if (d >= 0) rec(k + 1, acc + d);
        const tmp2 = perm[k]!;
        perm[k] = perm[i]!;
        perm[i] = tmp2;
      }
    };
    rec(0, 0);
    return best;
  }
  let sum = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (let g = 0; g < ctx.goals.length; g++) {
      const d = ctx.goalDist[g]![b]!;
      if (d >= 0 && d < m) m = d;
    }
    if (m === Infinity) return Infinity;
    sum += m;
  }
  return sum;
}

// 2x2 deadlock: a 2x2 block of walls/boxes holding a box off its goal can
// never move again. Only blocks containing `cell` can have changed.
function is2x2Dead(ctx: SearchCtx, boxes: number[], boxSet: Set<number>, cell: number): boolean {
  const { w, h } = ctx;
  const cx = cell % w;
  const cy = Math.floor(cell / w);
  for (let ox = -1; ox <= 0; ox++) {
    for (let oy = -1; oy <= 0; oy++) {
      const x0 = cx + ox;
      const y0 = cy + oy;
      if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;
      let allBlocked = true;
      let hasOffGoalBox = false;
      for (let yy = 0; yy < 2; yy++) {
        for (let xx = 0; xx < 2; xx++) {
          const c = (y0 + yy) * w + (x0 + xx);
          const isBox = boxSet.has(c);
          if (!isBox && ctx.walkable[c]) {
            allBlocked = false;
            break;
          }
          if (isBox && !ctx.goalSet.has(c)) hasOffGoalBox = true;
        }
        if (!allBlocked) break;
      }
      if (allBlocked && hasOffGoalBox) return true;
    }
  }
  return false;
}

// ---- Push-space A* (greedy=false) or best-first (greedy=true) -------------
interface SearchNode {
  boxes: number[];
  player: number; // canonical: smallest cell of the player's reachable region
  parent: number;
  from: number;
  to: number;
  dir: number;
  g: number;
}

function stateKey(boxes: number[], player: number): string {
  return boxes.join(',') + '|' + player;
}

// Canonical player cell: smallest cell of the region reachable from `player`
// over walkable cells with `boxes` as obstacles.
function canonPlayer(ctx: SearchCtx, player: number, boxSet: Set<number>): number {
  const { w, h, walkable } = ctx;
  let min = player;
  const seen = new Set<number>([player]);
  const q: number[] = [player];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    if (c < min) min = c;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!walkable[n] || boxSet.has(n) || seen.has(n)) continue;
      seen.add(n);
      q.push(n);
    }
  }
  return min;
}

// Reachable region of the player (for push generation).
function playerRegion(ctx: SearchCtx, player: number, boxSet: Set<number>): Set<number> {
  const { w, h, walkable } = ctx;
  const region = new Set<number>([player]);
  const q: number[] = [player];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!walkable[n] || boxSet.has(n) || region.has(n)) continue;
      region.add(n);
      q.push(n);
    }
  }
  return region;
}

function solvePush(ctx: SearchCtx, boxes0: number[], startPlayer: number, cap: number, greedy: boolean): Push[] | null {
  const { w, h } = ctx;
  const startBoxes = [...boxes0].sort((a, b) => a - b);
  if (startBoxes.every((b) => ctx.goalSet.has(b))) return [];
  const startSet = new Set(startBoxes);
  const startCanon = canonPlayer(ctx, startPlayer, startSet);
  const h0 = heuristic(ctx, startBoxes);
  if (h0 === Infinity) return null;

  const nodes: SearchNode[] = [
    { boxes: startBoxes, player: startCanon, parent: -1, from: -1, to: -1, dir: -1, g: 0 },
  ];
  const best = new Map<string, number>([[stateKey(startBoxes, startCanon), 0]]);
  // Heap entries [priority, g, nodeIdx]; priority = h (greedy) or g+h (A*).
  // Tie-break: deeper g first, then earlier node (fully deterministic).
  const heap: [number, number, number][] = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  heapPush([greedy ? h0 : h0, 0, 0]);

  let expanded = 0;
  while (heap.length > 0 && expanded < cap) {
    const [, g, idx] = heapPop();
    const node = nodes[idx]!;
    if (node.g !== g) continue;
    if (best.get(stateKey(node.boxes, node.player)) !== g) continue;
    expanded++;
    const boxSet = new Set(node.boxes);
    const region = playerRegion(ctx, node.player, boxSet);
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const from = node.boxes[bi]!;
      const fx = from % w;
      const fy = Math.floor(from / w);
      for (let d = 0; d < 4; d++) {
        const tx = fx + DX[d]!;
        const ty = fy + DY[d]!;
        if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
        const to = ty * w + tx;
        if (!ctx.boxCells[to] || boxSet.has(to)) continue;
        const sx = fx - DX[d]!;
        const sy = fy - DY[d]!;
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        const stand = sy * w + sx;
        if (!ctx.walkable[stand] || !region.has(stand)) continue;
        const nb = node.boxes.slice();
        nb[bi] = to;
        nb.sort((a, b) => a - b);
        const nbSet = new Set(nb);
        // Dead-square prune (skip boxes already on goals).
        let dead = false;
        for (const b of nb) {
          if (!ctx.goalSet.has(b) && !ctx.live[b]) {
            dead = true;
            break;
          }
        }
        if (dead) continue;
        if (is2x2Dead(ctx, nb, nbSet, to)) continue;
        // Goal test on generation.
        if (nb.every((b) => ctx.goalSet.has(b))) {
          const pushes: Push[] = [{ from, to, dir: d }];
          let t = idx;
          while (t > 0) {
            const nd = nodes[t]!;
            pushes.push({ from: nd.from, to: nd.to, dir: nd.dir });
            t = nd.parent;
          }
          pushes.reverse();
          return pushes;
        }
        const hNew = heuristic(ctx, nb);
        if (hNew === Infinity) continue;
        const gNew = g + 1;
        const newCanon = canonPlayer(ctx, from, nbSet);
        const k = stateKey(nb, newCanon);
        const prevBest = best.get(k);
        if (prevBest !== undefined && prevBest <= gNew) continue;
        best.set(k, gNew);
        const ni = nodes.length;
        nodes.push({ boxes: nb, player: newCanon, parent: idx, from, to, dir: d, g: gNew });
        heapPush([greedy ? hNew : gNew + hNew, gNew, ni]);
      }
    }
  }
  return null;
}

// Expand a push list into player moves (lowercase walks, uppercase pushes).
// Walkable = room cells + entry; `boxes0` and `player0` are the state before
// the first push. Returns null if a walk is impossible (should not happen).
function pushesToMoves(ctx: SearchCtx, boxes0: number[], player0: number, pushes: Push[]): string | null {
  const { w, h } = ctx;
  let moves = '';
  let boxes = [...boxes0].sort((a, b) => a - b);
  let player = player0;
  for (const p of pushes) {
    const stand = p.from - (DY[p.dir]! * w + DX[p.dir]!);
    const path = bfsPath(player, stand, ctx.walkable, new Set(boxes), w, h);
    if (path === null) return null;
    moves += pathLetters(player, path, w);
    moves += LETTER[p.dir]!.toUpperCase();
    const bi = boxes.indexOf(p.from);
    boxes[bi] = p.to;
    boxes.sort((a, b) => a - b);
    player = p.from;
  }
  return moves;
}

// ---- Whole-level fallback (used only if decomposition is invalid) --------
function solveWholeLevel(lv: Level): string {
  const allCells: number[] = [];
  for (let i = 0; i < lv.w * lv.h; i++) if (!lv.wall[i]) allCells.push(i);
  const ctx = buildCtx(lv.w, lv.h, lv.wall, allCells, -1, lv.goals);
  let pushes = solvePush(ctx, lv.boxes, lv.player, WHOLE_CAP, false);
  if (!pushes) pushes = solvePush(ctx, lv.boxes, lv.player, WHOLE_CAP, true);
  if (!pushes) return '';
  return pushesToMoves(ctx, lv.boxes, lv.player, pushes) ?? '';
}

// ---- Per-level solver -----------------------------------------------------
function solveLevel(text: string): string {
  const lv = parse(text);
  const rooms = findRooms(lv);
  const roomBoxCount = rooms.reduce((a, r) => a + r.boxes.length, 0);
  const valid =
    rooms.length > 0 &&
    roomBoxCount === lv.boxes.length &&
    rooms.every((r) => r.boxes.length === r.goals.length && r.boxes.length >= 1 && r.boxes.length <= 5);
  if (!valid) return solveWholeLevel(lv);

  const boardWalkable: boolean[] = lv.wall.map((w) => !w);
  let moves = '';
  let boxes = [...lv.boxes].sort((a, b) => a - b);
  let player = lv.player;
  for (const room of rooms) {
    // Walk through the hallway to this room's entry cell.
    const path = bfsPath(player, room.entry, boardWalkable, new Set(boxes), lv.w, lv.h);
    if (path === null) continue;
    moves += pathLetters(player, path, lv.w);
    player = room.entry;
    // Solve this room starting from the entry cell.
    const roomBoxes = boxes.filter((b) => room.cellSet.has(b));
    const ctx = buildCtx(lv.w, lv.h, lv.wall, room.cells, room.entry, room.goals);
    let pushes = solvePush(ctx, roomBoxes, player, ROOM_CAP, false);
    if (!pushes) pushes = solvePush(ctx, roomBoxes, player, ROOM_CAP, true);
    if (!pushes) continue;
    const roomMoves = pushesToMoves(ctx, roomBoxes, player, pushes);
    if (roomMoves === null) continue;
    moves += roomMoves;
    // Apply the pushes to the global box list; the player ends on the
    // last pushed box's old cell.
    for (const p of pushes) {
      const bi = boxes.indexOf(p.from);
      boxes[bi] = p.to;
      player = p.from;
    }
    boxes.sort((a, b) => a - b);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
