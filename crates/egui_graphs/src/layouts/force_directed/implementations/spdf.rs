// sfdp_agui_graph.rs
// SFDP-style scalable force-directed layout adapted for embedding into the
// agui_graph library. This file exposes a ForceAlgorithm implementation that
// uses a multilevel coarse-to-fine approach and a Barnes–Hut quadtree for
// fast repulsive force approximation.
//
// Notes:
// - This is a pragmatic adaptation of a single-file SFDP-like implementation
//   into the API shape used by agui_graph (see example in the prompt).
// - It constructs a lightweight internal graph representation from the
//   provided `Graph` wrapper (node index -> 0..n-1 mapping) so we can
//   coarsen without needing to instantiate new `Graph` objects.
// - Tunable parameters live in SfdfpState. State is serializable.
// - Parallelization is used via rayon for large node counts.
//
// To include in your crate, place this file under layouts/algorithms and add
// a `mod` entry in the parent module.

use crate::layouts::layout::AnimatedState;
use crate::layouts::LayoutState;
use crate::{DisplayEdge, DisplayNode, ForceAlgorithm, Graph};
use egui::{Rect, Vec2};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};
use petgraph::{stable_graph::NodeIndex, EdgeType};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SfdpState {
    pub is_running: bool,
    pub dt: f32,
    pub epsilon: f32,
    pub damping: f32,
    pub max_step: f32,
    pub k_scale: f32,
    pub c_attract: f32,
    pub c_repulse: f32,
    /// Barnes-Hut opening threshold (theta); smaller = more accurate slower
    pub theta: f32,
    /// Number of coarsening levels (max)
    pub max_levels: usize,
    /// Iterations per level (coarsest..finest). Will be scaled automatically.
    pub base_iters: usize,
    #[serde(skip)]
    pub last_avg_displacement: Option<f32>,
    pub step_count: u64,
}

impl LayoutState for SfdpState {}

impl Default for SfdpState {
    fn default() -> Self {
        SfdpState {
            is_running: false,
            dt: 0.05,
            epsilon: 1e-3,
            damping: 0.6,
            max_step: 20.0,
            k_scale: 1.0,
            c_attract: 1.0,
            c_repulse: 1.0,
            theta: 0.5,
            max_levels: 8,
            base_iters: 40,
            last_avg_displacement: None,
            step_count: 0,
        }
    }
}

impl AnimatedState for SfdpState {
    fn is_running(&self) -> bool {
        self.is_running
    }
    fn set_running(&mut self, v: bool) {
        self.is_running = v;
    }
    fn last_avg_displacement(&self) -> Option<f32> {
        self.last_avg_displacement
    }
    fn set_last_avg_displacement(&mut self, v: Option<f32>) {
        self.last_avg_displacement = v;
    }
    fn step_count(&self) -> u64 {
        self.step_count
    }
    fn set_step_count(&mut self, v: u64) {
        self.step_count = v;
    }
}

#[derive(Debug, Default)]
pub struct Sfdp {
    state: SfdpState,
    // reusable displacement buffer
    scratch_disp: Vec<Vec2>,
}

impl Sfdp {
    pub fn from_state(state: SfdpState) -> Self {
        Self {
            state,
            scratch_disp: Vec::new(),
        }
    }
}

impl ForceAlgorithm for Sfdp {
    type State = SfdpState;

    fn from_state(state: Self::State) -> Self {
        Self::from_state(state)
    }

    fn step<N, E, Ty, Ix, Dn, De>(&mut self, g: &mut Graph<N, E, Ty, Ix, Dn, De>, view: Rect)
    where
        N: Sync + Clone,
        E: Sync + Clone,
        Ty: Sync + EdgeType,
        Ix: Sync + petgraph::csr::IndexType,
        Dn: Sync + DisplayNode<N, E, Ty, Ix>,
        De: Sync + DisplayEdge<N, E, Ty, Ix, Dn>,
    {
        if !self.state.is_running || g.node_count() == 0 {
            return;
        }

        // Prepare lightweight internal representation (0..n-1 indices)
        let indices: Vec<_> = g.g().node_indices().collect();
        let n = indices.len();
        if self.scratch_disp.len() == n {
            self.scratch_disp.fill(Vec2::ZERO);
        } else {
            self.scratch_disp.resize(n, Vec2::ZERO);
        }

        // Build adjacency + edge list for internal processing only once per step
        println!("build_internal_representation");
        let (rep, _) = build_internal_representation(g, &indices);

        // Prepare constants
        println!("prepare_constants");
        let Some(_) = prepare_constants(view, n, self.state.k_scale) else {
            return;
        };
        let theta = self.state.theta as f64;

        // Build multilevel graphs
        println!("Build multilevel graphs");
        let mut levels: Vec<GraphRep> = Vec::new();
        let mut mappings: Vec<Vec<usize>> = Vec::new();
        levels.push(rep.clone());
        for lvl in 1..self.state.max_levels {
            let (coarse, mapping) = coarsen(&levels[lvl - 1]);
            if coarse.n >= levels[lvl - 1].n || coarse.n < 2 {
                break;
            }
            levels.push(coarse);
            mappings.push(mapping);
            if levels.last().unwrap().n <= 50 {
                break;
            }
        }

        // positions per level
        println!("positions per level");
        let l = levels.len();
        let mut pos_levels: Vec<Vec<Vec2>> = Vec::with_capacity(l);
        // initialize finest level positions from the graph
        let mut finest_pos = vec![Vec2::ZERO; n];
        for (i, &node_idx) in indices.iter().enumerate() {
            let loc = g.g().node_weight(node_idx).unwrap().location();
            finest_pos[i] = Vec2::new(loc.x, loc.y);
        }
        pos_levels.push(finest_pos);
        // for coarser levels we'll create placeholders
        for lvl in 1..l {
            pos_levels.push(vec![Vec2::ZERO; levels[lvl].n]);
        }

        // initialize coarsest positions randomly if necessary
        println!(" initialize coarsest positions randomly if necessary");
        use rand::thread_rng;
        let mut rng = thread_rng();
        let coarsest = l - 1;
        if pos_levels[coarsest].iter().all(|p| p == &Vec2::ZERO) {
            let ncoarse = levels[coarsest].n;
            let radius = (ncoarse as f32).sqrt() * 10.0;
            for i in 0..ncoarse {
                let angle: f32 = rng.gen::<f32>() * std::f32::consts::TAU;
                pos_levels[coarsest][i] = Vec2::new(radius * angle.cos(), radius * angle.sin());
            }
        }

        // Run multilevel from coarse to fine
        println!(" Run multilevel from coarse to fine");
        for lvl_rev in (0..l).rev() {
            let lvl = lvl_rev; // current level index
            let graph_rep = &levels[lvl];
            // choose iterations scaled by level (coarser -> more iterations)
            let iters = ((self.state.base_iters as f32) / (1.0 + (l - 1 - lvl) as f32 * 0.5))
                .max(5.0) as usize;
            // area and k for this level
            let area = (graph_rep.n as f32).sqrt() * 100.0;
            let k_level = ((area * area) / (graph_rep.n as f32)).sqrt() * self.state.k_scale;

            // relax with BH on this level
            relax_barnes_hut_level(
                &mut pos_levels[lvl],
                graph_rep,
                iters,
                area as f64,
                k_level as f64,
                theta,
                self.state.c_attract as f64,
                self.state.c_repulse as f64,
                self.state.epsilon as f64,
                self.state.damping as f64,
                self.state.max_step as f64,
            );

            // project to finer level if exists
            if lvl > 0 {
                let mapping = &mappings[lvl - 1];
                let finer_n = levels[lvl - 1].n;
                pos_levels[lvl - 1].fill(Vec2::ZERO);
                for i in 0..finer_n {
                    let ci = mapping[i];
                    // small jitter
                    let jitter = 1e-3f32;
                    let jitter_x: f32 = (rng.gen::<f32>() - 0.5) * jitter;
                    let jitter_y: f32 = (rng.gen::<f32>() - 0.5) * jitter;
                    pos_levels[lvl - 1][i] = pos_levels[lvl][ci] + Vec2::new(jitter_x, jitter_y);
                }
            }
        }

        // After finishing, pos_levels[0] holds final positions for nodes in internal order
        // Apply displacements to real graph nodes while respecting dt/damping/max_step
        println!(
            "After finishing, pos_levels[0] holds final positions for nodes in internal order"
        );
        let final_positions = &pos_levels[0];
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (i, &node_idx) in indices.iter().enumerate() {
            let old = g.g().node_weight(node_idx).unwrap().location();
            let new = final_positions[i];
            // compute step vector
            let delta = new - Vec2::new(old.x, old.y);
            let mut step = delta * self.state.dt * self.state.damping;
            let len = step.length();
            if len > self.state.max_step {
                step = step.normalized() * self.state.max_step;
            }
            let updated = old + step;
            if updated.x.is_finite() && updated.y.is_finite() {
                g.g_mut()
                    .node_weight_mut(node_idx)
                    .unwrap()
                    .set_location(updated);
                sum += len.min(self.state.max_step);
                count += 1;
            }
        }

        let avg = if count == 0 {
            None
        } else {
            Some(sum / count as f32)
        };
        self.state.last_avg_displacement = avg;
        self.state.step_count += 1;
    }

    fn state(&self) -> Self::State {
        self.state.clone()
    }
}

// ----------------- Internal lightweight graph rep -----------------

#[derive(Clone, Debug)]
struct GraphRep {
    n: usize,
    edges: Vec<(usize, usize)>,
    adj: Vec<Vec<usize>>,
}

fn build_internal_representation<N, E, Ty, Ix, Dn, De>(
    g: &Graph<N, E, Ty, Ix, Dn, De>,
    indices: &[NodeIndex<Ix>],
) -> (GraphRep, Vec<usize>)
where
    Ty: EdgeType,
    Ix: petgraph::csr::IndexType,
    N: Sync + Clone,
    E: Sync + Clone,
    Dn: Sync + DisplayNode<N, E, Ty, Ix>,
    De: Sync + DisplayEdge<N, E, Ty, Ix, Dn>,
{
    // map NodeIndex -> 0..n-1
    let mut index_map = vec![usize::MAX; g.g().node_bound()];
    for (i, &idx) in indices.iter().enumerate() {
        index_map[idx.index()] = i;
    }
    let n = indices.len();
    let mut adj = vec![Vec::new(); n];
    let mut edges = Vec::new();
    for edge in g.g().edge_references() {
        let u = edge.source();
        let v = edge.target();
        let iu = index_map[u.index()];
        let iv = index_map[v.index()];
        if iu >= n || iv >= n {
            continue;
        }
        if iu == iv {
            continue;
        }
        adj[iu].push(iv);
        adj[iv].push(iu);
        let a = iu.min(iv);
        let b = iu.max(iv);
        edges.push((a, b));
    }
    // deduplicate edges
    edges.sort_unstable();
    edges.dedup();
    for v in &mut adj {
        v.sort_unstable();
        v.dedup();
    }
    (GraphRep { n, edges, adj }, index_map)
}

// Greedy maximal matching coarsening (returns coarse graph and mapping fine->coarse)
fn coarsen(g: &GraphRep) -> (GraphRep, Vec<usize>) {
    let n = g.n;
    let mut matched = vec![false; n];
    let mut mapping = vec![usize::MAX; n];
    let mut new_index = 0usize;
    for u in 0..n {
        if matched[u] {
            continue;
        }
        let mut found = None;
        for &v in &g.adj[u] {
            if !matched[v] {
                found = Some(v);
                break;
            }
        }
        if let Some(v) = found {
            matched[u] = true;
            matched[v] = true;
            mapping[u] = new_index;
            mapping[v] = new_index;
            new_index += 1;
        } else {
            matched[u] = true;
            mapping[u] = new_index;
            new_index += 1;
        }
    }
    // build coarse edges
    let mut edge_set = HashSet::new();
    for &(u, v) in &g.edges {
        let cu = mapping[u];
        let cv = mapping[v];
        if cu != cv {
            let a = cu.min(cv);
            let b = cu.max(cv);
            edge_set.insert((a, b));
        }
    }
    let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    edges.sort_unstable();
    // build adj
    let mut adj = vec![Vec::new(); new_index];
    for &(u, v) in &edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    (
        GraphRep {
            n: new_index,
            edges,
            adj,
        },
        mapping,
    )
}

// ----------------- Barnes-Hut quadtree -----------------
#[derive(Debug)]
struct Quad {
    center: Vec2,
    half: f64,
    mass: f64,
    mass_pos: Vec2,
    children: [Option<Box<Quad>>; 4],
    point_index: Option<usize>,
}

impl Quad {
    fn new(center: Vec2, half: f64) -> Self {
        Quad {
            center,
            half,
            mass: 0.0,
            mass_pos: Vec2::ZERO,
            children: [None, None, None, None],
            point_index: None,
        }
    }

    fn insert(&mut self, pos: Vec2, index: usize) {
        if self.mass == 0.0 {
            self.mass = 1.0;
            self.mass_pos = pos;
            self.point_index = Some(index);
            return;
        }
        // If leaf with a point, push it down
        if let Some(old_idx) = self.point_index.take() {
            let old_pos = self.mass_pos;
            let child_idx_old = self.child_index_for(old_pos);
            if self.children[child_idx_old].is_none() {
                self.children[child_idx_old] = Some(Box::new(Quad::new(
                    self.child_center(child_idx_old),
                    self.half / 2.0,
                )));
            }
            self.children[child_idx_old]
                .as_mut()
                .unwrap()
                .insert(old_pos, old_idx);
        }
        // insert new point
        let child_idx = self.child_index_for(pos);
        if self.children[child_idx].is_none() {
            self.children[child_idx] = Some(Box::new(Quad::new(
                self.child_center(child_idx),
                self.half / 2.0,
            )));
        }
        self.children[child_idx]
            .as_mut()
            .unwrap()
            .insert(pos, index);
        // update mass center
        self.mass += 1.0;
        let prev = self.mass_pos * ((self.mass - 1.0) as f32);
        let combined = prev + pos;
        self.mass_pos = combined * (1.0 / (self.mass as f32));
    }

    fn child_index_for(&self, p: Vec2) -> usize {
        let left = p.x < self.center.x;
        let top = p.y < self.center.y;
        match (left, top) {
            (true, true) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (false, false) => 3,
        }
    }

    fn child_center(&self, idx: usize) -> Vec2 {
        let quarter = (self.half as f32) / 2.0;
        match idx {
            0 => self.center + Vec2::new(-(quarter), -(quarter)),
            1 => self.center + Vec2::new(quarter, -quarter),
            2 => self.center + Vec2::new(-quarter, quarter),
            3 => self.center + Vec2::new(quarter, quarter),
            _ => unreachable!(),
        }
    }

    // returns repulsive force vector acting on `pos`
    fn apply_repulsion(&self, pos: Vec2, theta: f64, k: f64) -> Vec2 {
        println!("apply_repulsion");
        if self.mass == 0.0 {
            return Vec2::ZERO;
        }
        let dx = self.mass_pos.x - pos.x;
        let dy = self.mass_pos.y - pos.y;
        let dist2 = (dx * dx + dy * dy) as f64 + 1e-9;
        let dist = dist2.sqrt();
        let size = (self.half * 2.0) as f64;
        if size / dist < theta || self.children.iter().all(|c| c.is_none()) {
            // approximate as single mass
            // repulsive ~ k^2 * mass / dist^2
            let force = (k * k) / dist2 * (self.mass as f64);
            let fx = -(dx as f64) / dist * force;
            let fy = -(dy as f64) / dist * force;
            return Vec2::new(fx as f32, fy as f32);
        }
        let mut res = Vec2::ZERO;
        for c in &self.children {
            if let Some(child) = c {
                let f = child.apply_repulsion(pos, theta, k);
                res += f;
            }
        }
        res
    }
}

// Build quad tree from positions
fn build_quad_tree(positions: &[Vec2]) -> Quad {
    println!("build_quad_tree");
    let mut minx = f32::INFINITY;
    let mut miny = f32::INFINITY;
    let mut maxx = f32::NEG_INFINITY;
    let mut maxy = f32::NEG_INFINITY;
    for p in positions.iter() {
        minx = minx.min(p.x);
        miny = miny.min(p.y);
        maxx = maxx.max(p.x);
        maxy = maxy.max(p.y);
    }
    if minx.is_infinite() || miny.is_infinite() {
        // no points — return a default quad
        return Quad::new(Vec2::ZERO, 1.0);
    }
    let cx = (minx + maxx) / 2.0;
    let cy = (miny + maxy) / 2.0;
    let half = ((maxx - minx).max(maxy - miny)) / 2.0 + 1e-6;
    let mut root = Quad::new(Vec2::new(cx, cy), half as f64);
    for (i, p) in positions.iter().enumerate() {
        root.insert(*p, i);
    }
    root
}

// Single-level relaxation using Barnes-Hut within a GraphRep
#[allow(clippy::too_many_arguments)]
fn relax_barnes_hut_level(
    positions: &mut [Vec2],
    graph: &GraphRep,
    iterations: usize,
    area: f64,
    k: f64,
    theta: f64,
    c_attract: f64,
    c_repulse: f64,
    epsilon: f64,
    damping: f64,
    max_step: f64,
) {
    let n = positions.len();
    let mut disp = vec![Vec2::ZERO; n];
    for iter in 0..iterations {
        // build tree
        let root = build_quad_tree(positions);
        // reset disp
        for d in &mut disp {
            *d = Vec2::ZERO;
        }

        // repulsive via BH (parallelize for big graphs)
        if n > 2000 {
            disp.par_iter_mut().enumerate().for_each(|(i, d)| {
                let f = root.apply_repulsion(positions[i], theta, k);
                *d += f * (c_repulse as f32);
            });
        } else {
            for i in 0..n {
                let f = root.apply_repulsion(positions[i], theta, k);
                disp[i] += f * (c_repulse as f32);
            }
        }

        // attractive forces along edges
        // use Fruchterman-Reingold style spring: force ~ dist^2 / k
        if graph.edges.len() < 50_000 {
            for &(u, v) in &graph.edges {
                let delta = positions[u] - positions[v];
                let dist = delta.length().max(epsilon as f32);
                let force = (dist * dist) as f64 / k; // scalar
                let vec = delta / dist * (force as f32);
                disp[u] -= vec * (c_attract as f32);
                disp[v] += vec * (c_attract as f32);
            }
        } else {
            // parallel edge processing aggregated into per-node accumulators
            // let mut adds = vec![Vec2::ZERO; n];
            // graph.edges.par_iter().for_each(|&(u, v)| {
            //     let delta = positions[u] - positions[v];
            //     let dist = delta.length().max(epsilon as f32);
            //     let force = (dist * dist) as f64 / k;
            //     let vec = delta / dist * (force as f32) * (c_attract as f32);
            //     // atomic updates would be ideal — we accumulate into local vec then reduce.
            //     // For simplicity in this example we'll do a coarse-grained reduction using chunking.
            //     // (In a production implementation, use atomic floats or per-thread buffers.)
            //     // Fallback: single-threaded accumulation if graph too big for safe parallel here.
            // });
            // Fallback: do single-threaded if we didn't compute adds
            for &(u, v) in &graph.edges {
                let delta = positions[u] - positions[v];
                let dist = delta.length().max(epsilon as f32);
                let force = (dist * dist) as f64 / k;
                let vec = delta / dist * (force as f32) * (c_attract as f32);
                disp[u] -= vec;
                disp[v] += vec;
            }
        }

        // apply displacements with temperature schedule
        let t = area * (1.0 - (iter as f64) / (iterations as f64));
        for i in 0..n {
            let d = disp[i].length();
            if d > 0.0 {
                let mut step = disp[i] * ((t.min(d as f64)) as f32 / d);
                // damp
                step *= damping as f32;
                let step_len = step.length();
                if step_len > max_step as f32 {
                    step = step.normalized() * (max_step as f32);
                }
                positions[i] += step;
            }
        }
    }
}

// prepare_constants compatible with agui_graph helper in example
pub(crate) fn prepare_constants(canvas: Rect, node_count: usize, k_scale: f32) -> Option<f32> {
    if node_count == 0 {
        return None;
    }
    let n = node_count as f32;
    let area = canvas.area().max(1.0);
    let k_ideal = (area / n).sqrt();
    let k = k_ideal * k_scale;
    if !k.is_finite() {
        return None;
    }
    Some(k)
}

// End of file
