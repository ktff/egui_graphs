use crate::layouts::layout::AnimatedState;
use crate::layouts::LayoutState;
use crate::{DisplayEdge, DisplayNode, ForceAlgorithm, Graph};
use egui::{Rect, Vec2};
use petgraph::{csr::IndexType, stable_graph::NodeIndex, EdgeType};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FruchtermanReingoldState {
    pub is_running: bool,
    pub dt: f32,
    pub epsilon: f32,
    pub damping: f32,
    pub max_step: f32,
    pub k_scale: f32,
    pub c_attract: f32,
    pub c_repulse: f32,
    #[serde(skip)]
    pub last_avg_displacement: Option<f32>,
    /// Total number of simulation steps executed.
    pub step_count: u64,
}

impl LayoutState for FruchtermanReingoldState {}

impl Default for FruchtermanReingoldState {
    fn default() -> Self {
        FruchtermanReingoldState {
            is_running: true,
            dt: 0.05,
            epsilon: 1e-3,
            damping: 0.3,
            max_step: 10.0,
            k_scale: 1.0,
            c_attract: 1.0,
            c_repulse: 1.0,
            last_avg_displacement: None,
            step_count: 0,
        }
    }
}

impl FruchtermanReingoldState {
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        is_running: bool,
        dt: f32,
        epsilon: f32,
        damping: f32,
        max_step: f32,
        k_scale: f32,
        c_attract: f32,
        c_repulse: f32,
    ) -> Self {
        Self {
            is_running,
            dt,
            epsilon,
            damping,
            max_step,
            k_scale,
            c_attract,
            c_repulse,
            last_avg_displacement: None,
            step_count: 0,
        }
    }
}

// Step counting is provided via AnimatedState default methods and field in this state.

impl AnimatedState for FruchtermanReingoldState {
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
pub struct FruchtermanReingold {
    state: FruchtermanReingoldState,
    // Reusable displacement buffer to avoid per-frame allocations
    scratch_disp: Vec<Vec2>,
}

impl FruchtermanReingold {
    pub fn from_state(state: FruchtermanReingoldState) -> Self {
        Self {
            state,
            scratch_disp: Vec::new(),
        }
    }
}

impl ForceAlgorithm for FruchtermanReingold {
    type State = FruchtermanReingoldState;

    fn from_state(state: Self::State) -> Self {
        Self {
            state,
            scratch_disp: Vec::new(),
        }
    }

    fn step<N, E, Ty, Ix, Dn, De>(&mut self, g: &mut Graph<N, E, Ty, Ix, Dn, De>, view: Rect)
    where
        N: Sync + Clone,
        E: Sync + Clone,
        Ty: Sync + EdgeType,
        Ix: Sync + IndexType,
        Dn: Sync + DisplayNode<N, E, Ty, Ix>,
        De: Sync + DisplayEdge<N, E, Ty, Ix, Dn>,
    {
        if !self.state.is_running || g.node_count() == 0 {
            return;
        }

        let params = &self.state;
        // Always compute k from the viewport area for stability and simplicity.
        let Some(k) = prepare_constants(view, g.node_count(), params.k_scale) else {
            return;
        };

        let indices: Vec<_> = g.g().node_indices().collect();
        // Ensure scratch buffer is sized and zeroed
        if self.scratch_disp.len() == indices.len() {
            self.scratch_disp.fill(Vec2::ZERO);
        } else {
            self.scratch_disp.resize(indices.len(), Vec2::ZERO);
        }

        compute_repulsion(
            g,
            &indices,
            &mut self.scratch_disp,
            k,
            params.epsilon,
            params.c_repulse,
        );
        compute_attraction(
            g,
            &indices,
            &mut self.scratch_disp,
            k,
            params.epsilon,
            params.c_attract,
        );
        let avg = apply_displacements(
            g,
            &indices,
            &self.scratch_disp,
            params.dt,
            params.damping,
            params.max_step,
        );
        self.state.last_avg_displacement = avg;
        self.state.set_step_count(self.state.step_count + 1);
    }

    fn state(&self) -> Self::State {
        self.state.clone()
    }
}

pub(crate) fn prepare_constants(canvas: Rect, node_count: usize, k_scale: f32) -> Option<f32> {
    if node_count == 0 {
        return None;
    }
    let n = node_count as f32;
    let area = canvas.area().max(1.0);
    let k_ideal = (area / n).sqrt(); // ideal edge length
    let k = k_ideal * k_scale;
    if !k.is_finite() {
        return None;
    }
    Some(k)
}

pub(crate) fn compute_repulsion<N: Sync, E: Sync, Ty: Sync, Ix: Sync, Dn: Sync, De: Sync>(
    g: &Graph<N, E, Ty, Ix, Dn, De>,
    indices: &[NodeIndex<Ix>],
    disp: &mut [Vec2],
    k: f32,
    epsilon: f32,
    c_repulse: f32,
) where
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
    Dn: DisplayNode<N, E, Ty, Ix>,
    De: DisplayEdge<N, E, Ty, Ix, Dn>,
{
    if indices.len() < 1000 {
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let (idx_i, idx_j) = (indices[i], indices[j]);
                let delta = g.g().node_weight(idx_i).unwrap().location()
                    - g.g().node_weight(idx_j).unwrap().location();
                let distance = delta.length().max(epsilon);
                let force = c_repulse * (k * k) / distance;
                let dir = delta / distance;
                disp[i] += dir * force;
                disp[j] -= dir * force;
            }
        }
    } else {
        let computed = (0..indices.len())
            .into_par_iter()
            .map(|i| {
                let mut disp = Vec::with_capacity(indices.len());
                let mut rand = rand::rng();
                disp.resize(indices.len(), Vec2::default());
                let range = (i + 1)..indices.len();
                for _ in 0..1000.min(range.end - range.start) {
                    let j = rand.random_range(range.clone());
                    let (idx_i, idx_j) = (indices[i], indices[j]);
                    let delta = g.g().node_weight(idx_i).unwrap().location()
                        - g.g().node_weight(idx_j).unwrap().location();
                    let distance = delta.length().max(epsilon);
                    let force = c_repulse * (k * k) / distance;
                    let dir = delta / distance;
                    disp[i] += dir * force;
                    disp[j] -= dir * force;
                }
                disp
            })
            .reduce(
                || Vec::new(),
                |mut acc, add| {
                    if acc.is_empty() {
                        add
                    } else {
                        acc.iter_mut().zip(add).for_each(|(acc, add)| *acc += add);
                        acc
                    }
                },
            );

        disp.iter_mut()
            .zip(computed)
            .for_each(|(acc, add)| *acc += add);
    }
}

pub(crate) fn compute_attraction<N, E, Ty, Ix, Dn, De>(
    g: &Graph<N, E, Ty, Ix, Dn, De>,
    indices: &[NodeIndex<Ix>],
    disp: &mut [Vec2],
    k: f32,
    epsilon: f32,
    c_attract: f32,
) where
    N: Sync + Clone,
    E: Sync + Clone,
    Ty: Sync + EdgeType,
    Ix: Sync + IndexType,
    Dn: Sync + DisplayNode<N, E, Ty, Ix>,
    De: Sync + DisplayEdge<N, E, Ty, Ix, Dn>,
{
    if indices.len() < 5000 {
        for (vec_pos, &idx) in indices.iter().enumerate() {
            let loc = g.g().node_weight(idx).unwrap().location();
            for nbr in g.g().neighbors_undirected(idx) {
                let delta = g.g().node_weight(nbr).unwrap().location() - loc;
                let distance = delta.length().max(epsilon);
                let force = c_attract * (distance * distance) / k;
                disp[vec_pos] += (delta / distance) * force;
            }
        }
    } else {
        indices
            .par_iter()
            .enumerate()
            .map(|(vec_pos, &idx)| {
                let mut disp = Vec2::default();
                let loc = g.g().node_weight(idx).unwrap().location();
                for nbr in g.g().neighbors_undirected(idx) {
                    let delta = g.g().node_weight(nbr).unwrap().location() - loc;
                    let distance = delta.length().max(epsilon);
                    let force = c_attract * (distance * distance) / k;
                    disp += (delta / distance) * force;
                }
                (vec_pos, disp)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(i, add)| disp[i] += add);
    }
}

pub(crate) fn apply_displacements<N, E, Ty, Ix, Dn, De>(
    g: &mut Graph<N, E, Ty, Ix, Dn, De>,
    indices: &[NodeIndex<Ix>],
    disp: &[Vec2],
    dt: f32,
    damping: f32,
    max_step: f32,
) -> Option<f32>
where
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
    Dn: DisplayNode<N, E, Ty, Ix>,
    De: DisplayEdge<N, E, Ty, Ix, Dn>,
{
    if indices.is_empty() {
        return Some(0.0);
    }
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for (vec_pos, &idx) in indices.iter().enumerate() {
        let mut step = disp[vec_pos] * dt * damping;
        let len = step.length();
        if len > max_step {
            step = step.normalized() * max_step;
        }
        let loc = g.g().node_weight(idx).unwrap().location();
        let new_loc = loc + step;
        if !new_loc.x.is_finite() || !new_loc.y.is_finite() {
            continue;
        }
        g.g_mut()
            .node_weight_mut(idx)
            .unwrap()
            .set_location(new_loc);
        sum += len.min(max_step);
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{to_graph, DefaultEdgeShape, DefaultNodeShape};
    use egui::{Pos2, Rect};
    use petgraph::stable_graph::StableGraph;

    fn empty_ui_rect() -> Rect {
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 1000.0))
    }

    fn make_graph(
        num: usize,
    ) -> Graph<
        (),
        (),
        petgraph::Directed,
        petgraph::stable_graph::DefaultIx,
        DefaultNodeShape,
        DefaultEdgeShape,
    > {
        let mut g: StableGraph<(), ()> = StableGraph::default();
        for _ in 0..num {
            g.add_node(());
        }
        let mut graph = to_graph(&g);
        let node_indices: Vec<_> = graph.g().node_indices().collect();
        for (i, idx) in node_indices.iter().enumerate() {
            let mut_loc = Pos2::new(i as f32 * 10.0, 0.0);
            graph
                .g_mut()
                .node_weight_mut(*idx)
                .unwrap()
                .set_location(mut_loc);
        }
        graph
    }

    #[test]
    fn repulsion_increases_distance() {
        let mut g = make_graph(2);
        let idxs: Vec<_> = g.g().node_indices().collect();
        g.g_mut()
            .node_weight_mut(idxs[0])
            .unwrap()
            .set_location(Pos2::new(0.0, 0.0));
        g.g_mut()
            .node_weight_mut(idxs[1])
            .unwrap()
            .set_location(Pos2::new(1.0, 0.0));
        let rect = empty_ui_rect();
        let params = FruchtermanReingoldState::default();
        let k = prepare_constants(rect, 2, params.k_scale).unwrap();
        let indices: Vec<_> = g.g().node_indices().collect();
        let mut disp = vec![Vec2::ZERO; indices.len()];
        compute_repulsion(&g, &indices, &mut disp, k, params.epsilon, params.c_repulse);
        apply_displacements(
            &mut g,
            &indices,
            &disp,
            params.dt,
            params.damping,
            params.max_step,
        );
        let a = g.g().node_weight(indices[0]).unwrap().location();
        let b = g.g().node_weight(indices[1]).unwrap().location();
        assert!((b.x - a.x).abs() > 1.0, "Nodes should move apart");
    }

    #[test]
    fn attraction_decreases_distance_when_far() {
        let mut g = make_graph(2);
        let idxs: Vec<_> = g.g().node_indices().collect();
        g.add_edge(idxs[0], idxs[1], ());
        g.g_mut()
            .node_weight_mut(idxs[0])
            .unwrap()
            .set_location(Pos2::new(0.0, 0.0));
        g.g_mut()
            .node_weight_mut(idxs[1])
            .unwrap()
            .set_location(Pos2::new(1200.0, 0.0));
        let rect = empty_ui_rect();
        let params = FruchtermanReingoldState::default();
        let k = prepare_constants(rect, 2, params.k_scale).unwrap();
        let indices: Vec<_> = g.g().node_indices().collect();
        let mut disp = vec![Vec2::ZERO; indices.len()];
        let start_dist = 1200.0;
        compute_repulsion(&g, &indices, &mut disp, k, params.epsilon, params.c_repulse);
        compute_attraction(&g, &indices, &mut disp, k, params.epsilon, params.c_attract);
        apply_displacements(
            &mut g,
            &indices,
            &disp,
            params.dt,
            params.damping,
            params.max_step,
        );
        let a = g.g().node_weight(indices[0]).unwrap().location();
        let b = g.g().node_weight(indices[1]).unwrap().location();
        let new_dist = (b - a).length();
        assert!(
            new_dist < start_dist,
            "Distance should shrink due to attraction"
        );
    }
}
