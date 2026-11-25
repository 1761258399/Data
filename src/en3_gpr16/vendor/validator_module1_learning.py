"""Shared causal event data, physical memory and GP algorithms, used by GUI and CLI."""

import csv, os, math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from scipy.optimize import minimize

def gpr_optimizer_global(obj_func, initial_theta, bounds):
    res = minimize(obj_func, initial_theta, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 50000, 'maxfun': 50000, 'ftol': 1e-09})
    return (res.x, res.fun)

P2_MEMORY_VERSION = 'causal_play_memory_v2'
P2_REPLAYABLE_VERSIONS = {'', 'causal_play_memory_v1', P2_MEMORY_VERSION}

def p2_ablate_features(features, variant='none'):
    """Project both training and online queries with the same frozen mechanism mask."""
    x = np.asarray(features, float).copy()
    if variant not in ('none', 'no_stop', 'no_dwell'):
        raise ValueError('Unknown physical-memory ablation: ' + str(variant))
    if variant != 'none' and x.shape[-1] != 60:
        raise ValueError('Physical-memory ablation requires 60 features')
    if variant == 'no_stop': x[..., 40:48] = 0.
    if variant == 'no_dwell':
        x[..., 52:56] = 0.
        x[..., 57] = 0.
        x[..., 59] = 0.
    return x

P2_DEFAULT_RADII_MM = (0.025, 0.1)

P2_FEATURE_COUNT = 60

P2_FEATURE_ORDER = ['q_m_i', 'q_m_im1', 'q_m_im2', 'candidate_u_i', 'u_im1', 'u_im2', 'tracking_error_i', 'reversal_i', 'memory_last_direction', 'memory_log_run_travel', 'memory_play_gap_small', 'memory_play_gap_large', 'memory_last_rate', 'memory_log_idle', 'memory_age_fraction', 'planned_settle_log_s', 'estimated_move_log_s', 'boundary_age_log_s']

def p2_move_duration(command, speed=5.0, acceleration=5.0):
    """Causal trapezoidal-motion proxy, not a future measured DONE duration."""
    u = np.asarray(command, dtype=float).reshape(4)
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite([speed, acceleration])) or speed <= 0 or (acceleration <= 0):
        raise ValueError('invalid motion-duration inputs')
    distance = float(np.max(np.abs(u)))
    return 2.0 * math.sqrt(distance / acceleration) if distance <= speed * speed / acceleration else distance / speed + speed / acceleration

class PhysicalEventMemory:
    """Command-driven play bank plus causal time/history summaries.

    The play state is a reproducible model state, not an independently
    measured mechanical backlash state. Reset initialization is explicit.
    Queries never advance it. Only qualified, completed transitions commit.
    """

    def __init__(self, radii=P2_DEFAULT_RADII_MM):
        self.radii = np.asarray(radii, dtype=float).reshape(2)
        if np.any(~np.isfinite(self.radii)) or np.any(self.radii <= 0):
            raise ValueError('play radii must be finite and positive')
        self.reset()

    def reset(self, boundary_time=None):
        self.direction = np.zeros(4)
        self.run_travel = np.zeros(4)
        self.gaps = np.zeros((2, 4))
        self.last_rate = np.zeros(4)
        self.last_active_done = np.full(4, np.nan)
        self.boundary_time = float(boundary_time) if boundary_time is not None else np.nan
        self.commits = 0

    def commit(self, command, send_time, done_time, boundary_time, qualified=True):
        if not qualified:
            self.reset(boundary_time)
            return False
        u = np.asarray(command, dtype=float).reshape(4)
        times = np.asarray([send_time, done_time, boundary_time], dtype=float)
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(times)) or (not send_time < done_time <= boundary_time) or (np.isfinite(self.boundary_time) and send_time < self.boundary_time - 1e-06):
            raise ValueError('memory commit is not a completed causal transition')
        active = np.abs(u) > 1e-08
        direction = np.sign(u)
        reversed_axis = active & (direction != self.direction)
        self.run_travel[reversed_axis] = 0.0
        self.run_travel[active] += np.abs(u[active])
        self.direction[active] = direction[active]
        self.gaps = np.clip(self.gaps + u[None, :], -self.radii[:, None], self.radii[:, None])
        self.last_rate[active] = u[active] / max(done_time - send_time, 1e-06)
        self.last_active_done[active] = done_time
        self.boundary_time = float(boundary_time)
        self.commits += 1
        return True

    def features(self, now, planned_settle_s, estimated_move_s):
        now = float(now)
        if not np.isfinite(now) or planned_settle_s < 0 or estimated_move_s < 0 or (not np.isfinite(planned_settle_s + estimated_move_s)):
            raise ValueError('non-finite or negative causal memory time')
        start = self.boundary_time if np.isfinite(self.boundary_time) else now
        if now < start - 1e-06:
            raise ValueError('query predates the accepted memory boundary')
        last_active = np.where(np.isfinite(self.last_active_done), self.last_active_done, start)
        idle = np.maximum(now - last_active, 0.0)
        return np.concatenate([self.direction, np.log1p(self.run_travel), self.gaps[0], self.gaps[1], self.last_rate, np.log1p(idle), [min(self.commits / 8.0, 1.0), math.log1p(planned_settle_s), math.log1p(estimated_move_s), math.log1p(max(now - start, 0.0))]])

def p2_causal_features(q_history, u_history, command_state, candidate, memory, query_time, settle_s, speed=5.0, acceleration=5.0):
    q = np.asarray(q_history, dtype=float).reshape(3, 4)
    previous = np.asarray(u_history, dtype=float).reshape(2, 4)
    u = np.asarray(candidate, dtype=float).reshape(4)
    qc = np.asarray(command_state, dtype=float).reshape(4)
    reversal = ((np.abs(u) > 0.0001) & (np.abs(previous[0]) > 0.0001) & (u * previous[0] < 0)).astype(float)
    base = np.concatenate([q[0], q[1], q[2], u, previous[0], previous[1], qc - q[0], reversal])
    tail = memory.features(query_time, settle_s, p2_move_duration(u, speed, acceleration))
    result = np.concatenate([base, tail])
    if result.shape != (P2_FEATURE_COUNT,) or not np.all(np.isfinite(result)):
        raise ValueError('physical-memory feature contract is invalid')
    return result

def p2_physical_basis(features, radii=P2_DEFAULT_RADII_MM, relaxation_s=1.0, ablation='none'):
    """Five displacement-level mean terms per axis, using pre-event state."""
    x = np.asarray(features, dtype=float)
    x = x.reshape(-1, P2_FEATURE_COUNT)
    if not np.all(np.isfinite(x)) or relaxation_s <= 0:
        raise ValueError('invalid physical mean inputs')
    radii = np.asarray(radii, dtype=float).reshape(2)
    u = x[:, 12:16]
    small, large = (x[:, 40:44], x[:, 44:48])
    loss_small = np.clip(small + u, -radii[0], radii[0]) - small
    loss_large = np.clip(large + u, -radii[1], radii[1]) - large
    idle = np.expm1(np.clip(x[:, 52:56], 0.0, 16.0))
    settle = np.expm1(np.clip(x[:, 57:58], 0.0, 16.0))
    relaxation = x[:, 32:36] * -np.expm1(-(idle + settle) / relaxation_s)
    basis = np.stack([u, loss_small, loss_large, relaxation, np.ones_like(u)], axis=2)
    if ablation == 'no_stop': basis[:, :, 1:3] = 0.
    elif ablation == 'no_dwell': basis[:, :, 3] = 0.
    elif ablation != 'none': raise ValueError('Unknown physical basis ablation')
    return basis

def p2_fit_physical_mean(features, targets, radii=P2_DEFAULT_RADII_MM, relaxation_s=1.0, ridge=0.0001, ablation='none'):
    """Bounded Huber-reweighted mean; the remaining error is learned by GP.

    Bounds constrain the physical prior only. They do not constrain the total
    GP prediction or imply torque passivity, stability or a safety theorem.
    """
    basis = p2_physical_basis(features, radii, relaxation_s, ablation)
    y = np.asarray(targets, dtype=float).reshape(-1, 4)
    if len(basis) != len(y) or not np.all(np.isfinite(y)):
        raise ValueError('invalid physical mean training targets')
    beta = np.zeros((4, 5))
    for axis in range(4):
        b = basis[:, axis, :]
        weights = np.ones(len(y))
        coeff = np.zeros(5)
        for _ in range(5):

            def objective(v):
                residual = b @ v - y[:, axis]
                value = float(np.sum(weights * residual ** 2) + ridge * (v @ v))
                gradient = 2.0 * (b.T @ (weights * residual) + ridge * v)
                return (value, gradient)
            result = minimize(objective, coeff, jac=True, method='SLSQP', bounds=[(0, 0.5), (0, 1), (0, 1), (-0.1, 0.1), (-0.1, 0.1)], constraints=[{'type': 'ineq', 'fun': lambda v: 1.0 - v[1] - v[2], 'jac': lambda v: np.array([0.0, -1.0, -1.0, 0.0, 0.0])}], options={'maxiter': 200, 'ftol': 1e-10})
            if not result.success or not np.all(np.isfinite(result.x)):
                raise ValueError('physical prior fitting failed: ' + str(result.message))
            coeff = result.x
            residual = y[:, axis] - b @ coeff
            scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 0.002)
            weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-12))
        beta[axis] = coeff
    return beta

def p2_physical_mean(features, coefficients, radii=P2_DEFAULT_RADII_MM, relaxation_s=1.0, ablation='none'):
    b = p2_physical_basis(features, radii, relaxation_s, ablation)
    beta = np.asarray(coefficients, dtype=float).reshape(4, 5)
    if not np.all(np.isfinite(beta)):
        raise ValueError('non-finite physical mean coefficients')
    return np.einsum('nak,ak->na', b, beta)

class TaskActiveEventSampler:
    """Finite candidate P1 collector with qualified-boundary information.

    Information matrices are whitened design proxies. Geometry parameters
    remain frozen during residual collection. They are not latent-state truth
    or a certificate for joint geometric/GP identifiability.
    """

    def __init__(self, seed, strategy='task_active', position_limit=10.0, increment_limit=0.5, command_budget=6000, time_budget_s=18000.0, candidates=40, settle_s=1.0, speed=5.0, acceleration=5.0, radii=P2_DEFAULT_RADII_MM):
        if strategy not in ('task_active', 'fisher', 'variance', 'random'):
            raise ValueError('unknown P1 acquisition strategy')
        values = [position_limit, increment_limit, time_budget_s, speed, acceleration]
        if not np.all(np.isfinite(values)) or min(values) <= 0 or command_budget < 1:
            raise ValueError('invalid acquisition budgets')
        self.rng = np.random.default_rng(int(seed))
        self.seed = int(seed)
        self.strategy = strategy
        self.limit = float(position_limit)
        self.step = float(increment_limit)
        self.budget = int(command_budget)
        self.time_budget = float(time_budget_s)
        self.candidates = min(max(int(candidates), 16), 128)
        self.settle_s = float(settle_s)
        self.speed = float(speed)
        self.acceleration = float(acceleration)
        self.memory = PhysicalEventMemory(radii)
        self.q_history = []
        self.u_history = []
        self.qc = None
        self.memory_command_state = None
        self.last_boundary_time = np.nan
        self.geom_info = np.zeros((4, 4))
        self.physical_info = np.zeros((4, 5, 5))
        self.cells = {}
        self.valid_transitions = 0
        self.invalid_boundaries = 0
        self.sent = 0
        self.started = None
        self.pending = None
        self.geometry_reference = None

    @staticmethod
    def _loggain(info, rows):
        regularized = np.asarray(info) + np.eye(info.shape[0])
        rows = np.asarray(rows, dtype=float).reshape(-1, info.shape[0])
        after = regularized + rows.T @ rows
        return float(np.linalg.slogdet(after)[1] - np.linalg.slogdet(regularized)[1])

    def _geometry_subspace(self, rows):
        prospective = self.geom_info + rows.T @ rows
        eigen, vectors = np.linalg.eigh(prospective)
        keep = eigen > max(float(np.max(eigen)) * 1e-06, 1e-09)
        return vectors[:, keep]

    def observe(self, qc, qm, boundary_time, valid, geometry_jacobian, send_time=np.nan, done_time=np.nan):
        qc = np.asarray(qc, dtype=float).reshape(4)
        qm = np.asarray(qm, dtype=float).reshape(4)
        boundary_time = float(boundary_time)
        if self.started is None:
            self.started = boundary_time
        accepted = bool(valid and np.all(np.isfinite(qc)) and np.all(np.isfinite(qm)))
        if not accepted:
            self.invalid_boundaries += 1
            self.q_history = []
            self.u_history = []
            self.qc = None
            self.memory_command_state = qc.copy() if np.isfinite(qc).all() else None
            self.last_boundary_time = boundary_time
            self.memory.reset(boundary_time)
            self.pending = None
            return False
        j = np.asarray(geometry_jacobian(qm), dtype=float).reshape(6, 4)
        if not np.all(np.isfinite(j)):
            raise ValueError('invalid geometry design Jacobian')
        self.geom_info += j.T @ j
        if self.geometry_reference is None:
            self.geometry_reference = j.copy()
        if self.memory_command_state is not None:
            command = qc - self.memory_command_state
            if np.isfinite(send_time) and np.isfinite(done_time) and (self.last_boundary_time <= send_time < done_time <= boundary_time) and (np.max(np.abs(command)) <= self.step + 1e-06):
                if self.pending is not None and self.pending.get('history_ready', False):
                    b = p2_physical_basis(self.pending['features'], self.memory.radii)[0]
                    for axis in range(4):
                        row = b[axis] / np.array([self.step, 0.05, 0.2, 1.0, 1.0])
                        self.physical_info[axis] += np.outer(row, row)
                self.memory.commit(command, send_time, done_time, boundary_time)
                self.u_history.insert(0, command.copy())
                self.u_history = self.u_history[:2]
                self.valid_transitions += 1
            else:
                self.q_history = []
                self.u_history = []
                self.memory.reset(boundary_time)
        else:
            self.memory.reset(boundary_time)
        self.q_history.insert(0, qm.copy())
        self.q_history = self.q_history[:3]
        self.qc = qc.copy()
        self.memory_command_state = qc.copy()
        self.last_boundary_time = boundary_time
        self.pending = None
        cell = tuple(np.clip(np.floor((qc + self.limit) / (2 * self.limit) * 6), 0, 5).astype(int))
        self.cells[cell] = self.cells.get(cell, 0) + 1
        return True

    def propose(self, current_command, now, geometry_jacobian, task_axis_jacobian, predictive_variance=None, blocked=None):
        if self.pending is not None:
            raise ValueError('an acquisition event is already pending')
        now = float(now)
        qc = np.asarray(current_command, dtype=float).reshape(4)
        if self.sent >= self.budget:
            return (None, 'command_budget')
        if self.started is not None and now - self.started >= self.time_budget:
            return (None, 'time_budget')
        if self.qc is None:
            return (None, 'qualified_boundary_required')
        candidates = []
        for axis in range(4):
            for value in (-self.step, -0.1 * self.step, 0.1 * self.step, self.step):
                u = np.zeros(4)
                u[axis] = value
                candidates.append(u)
        for _ in range(max(self.candidates - len(candidates), 0)):
            u = self.rng.uniform(-1.0, 1.0, 4)
            u *= self.step / max(float(np.max(np.abs(u))), 1e-12)
            u *= float(self.rng.choice([0.1, 0.4, 1.0]))
            candidates.append(u)
        ready = len(self.q_history) == 3 and len(self.u_history) == 2
        qh = list(self.q_history)
        uh = list(self.u_history)
        while len(qh) < 3:
            qh.append(qh[-1].copy())
        while len(uh) < 2:
            uh.append(np.zeros(4))
        options = []
        seen = set()
        time_rejected = False
        for u in candidates:
            target = np.round(np.clip(qc + u, -self.limit, self.limit), 4)
            u = target - qc
            token = tuple(target.tolist())
            if token in seen or np.max(np.abs(u)) < 1e-05:
                continue
            seen.add(token)
            if np.max(np.abs(u)) > self.step + 1e-06 or (blocked and blocked(target)):
                continue
            features = p2_causal_features(qh, uh, qc, u, self.memory, now, self.settle_s, self.speed, self.acceleration)
            j = np.asarray(geometry_jacobian(target), dtype=float).reshape(6, 4)
            task = np.asarray(task_axis_jacobian(target), dtype=float).reshape(6, 4)
            if not np.all(np.isfinite(j)) or not np.all(np.isfinite(task)):
                continue
            subspace = self._geometry_subspace(j)
            info = subspace.T @ self.geom_info @ subspace
            geom = self._loggain(info, j @ subspace) if subspace.shape[1] else 0.0
            b = p2_physical_basis(features, self.memory.radii)[0]
            physical = sum((self._loggain(self.physical_info[a], b[a] / np.array([self.step, 0.05, 0.2, 1.0, 1.0])) for a in range(4))) / 4.0
            before = np.linalg.inv(info + np.eye(len(info)))
            after = np.linalg.inv(info + np.eye(len(info)) + (j @ subspace).T @ (j @ subspace))
            reference = self.geometry_reference if self.geometry_reference is not None else j
            task_gain = max(float(np.trace(reference @ subspace @ (before - after) @ subspace.T @ reference.T)), 0.0)
            cell = tuple(np.clip(np.floor((target + self.limit) / (2 * self.limit) * 6), 0, 5).astype(int))
            novelty = 1.0 / (1.0 + self.cells.get(cell, 0))
            time_cost = p2_move_duration(u, self.speed, self.acceleration) + self.settle_s + 0.32
            if self.started is not None and now - self.started + time_cost > self.time_budget:
                time_rejected = True
                continue
            base = (geom + physical + 0.25 * novelty + 0.25 * task_gain) / max(time_cost, 0.05)
            options.append({'target': target, 'command': u, 'features': features, 'history_ready': ready, 'geometry_gain': geom, 'physical_gain': physical, 'task_gain': task_gain, 'novelty': novelty, 'estimated_cost_s': time_cost, 'geometry_candidate_rank': int(subspace.shape[1]), 'score': base, 'predictive_task_variance': 0.0, 'variance_available': False})
        if not options:
            return (None, 'time_budget' if time_rejected else 'no_eligible_candidate')
        if self.strategy == 'random':
            chosen = options[int(self.rng.integers(len(options)))]
        else:
            short = options if self.strategy in ('fisher', 'variance') else sorted(options, key=lambda x: x['score'], reverse=True)[:8]
            if ready and predictive_variance is not None and (self.strategy != 'fisher'):
                for option in short:
                    try:
                        var = np.asarray(predictive_variance(option['features']), dtype=float).reshape(4)
                        task = np.asarray(task_axis_jacobian(option['target']), dtype=float).reshape(6, 4)
                        if np.any(~np.isfinite(var)) or np.any(var < 0):
                            continue
                        option['predictive_task_variance'] = float(np.sum(np.sum(task ** 2, axis=0) * var))
                        option['variance_available'] = True
                    except (ValueError, FloatingPointError):
                        pass
            for option in short:
                if self.strategy == 'fisher':
                    option['score'] = option['geometry_gain'] + option['physical_gain']
                elif self.strategy == 'variance' and option['variance_available']:
                    option['score'] = option['predictive_task_variance']
                elif self.strategy == 'task_active':
                    option['score'] += 0.1 * math.log1p(option['predictive_task_variance']) / option['estimated_cost_s']
            if self.strategy == 'variance':
                available = [x for x in short if x['variance_available']]
                if ready and not available:
                    return (None, 'variance_model_unavailable')
                chosen = max(available or short, key=lambda x: x['score'])
            elif self.strategy == 'task_active' and ready:
                available = [x for x in short if x['variance_available']]
                if not available:return (None,'task_active_model_unavailable')
                chosen = max(available,key=lambda x:x['score'])
            else:
                chosen = max(short, key=lambda x: x['score'])
        chosen = dict(chosen)
        chosen['eligible_candidates'] = len(options)
        chosen['strategy'] = self.strategy
        chosen['selection_index'] = self.sent + 1
        chosen['query_monotonic'] = now
        chosen['effective_strategy'] = 'information_bootstrap' if self.strategy in ('variance','task_active') and (not chosen['variance_available']) else self.strategy
        return (chosen, '')

    def mark_sent(self, choice):
        if self.pending is not None:
            raise ValueError('pending P1 command cannot be overwritten')
        self.pending = dict(choice)
        self.sent += 1

    def diagnostics(self):
        eigen = np.linalg.eigvalsh(self.geom_info)
        cutoff = max(float(np.max(eigen)) * 1e-06, 1e-09)
        return {'seed': self.seed, 'strategy': self.strategy, 'commands_sent': self.sent, 'valid_transitions': self.valid_transitions, 'invalid_boundaries': self.invalid_boundaries, 'geometry_rank': int(np.count_nonzero(eigen > cutoff)), 'geometry_eigenvalues': eigen.tolist(), 'observed_command_cells': len(self.cells), 'command_budget': self.budget, 'time_budget_s': self.time_budget, 'information_semantics': 'whitened_design_proxy_fixed_geometry', 'geometry_subspace_relative_threshold': 1e-06, 'command_count_semantics': 'accepted_host_send_requests_not_verified_serial_writes'}

class EventLearning:

    def __init__(self, feature_version=9, **options):
        self.gpr_data_schema_version = 5
        self.gpr_model_format_version = int(feature_version)
        self.gpr_sample_feature_version = int(feature_version)
        self.gpr_command_timeout_s = 30.
        self.gpr_max_ndi_frame_age_s = .075
        self.gpr_ndi_edge_guard_boundaries = 2
        self.gpr_ndi_max_boundary_jump_mm = .4
        self.gpr_reversal_command_epsilon = 1e-4
        self.gpr_sweep_settle_ms = 1000.
        self.gpr_training_max_increment = .5
        self.p2_play_radii_mm = list(P2_DEFAULT_RADII_MM)
        self.p2_relaxation_s = 1.
        self.p2_ablation = 'none'
        self.gpr_training_family = 'gpr'
        self.gpr_model_family = 'gpr'
        self.gpr_training_n_restarts = 0
        self.gpr_training_optimize = True
        self.gpr_training_seed = 0
        self.gpr_std_calibration_scale = np.ones(4)
        for key, value in options.items():
            if not hasattr(self, key):
                raise ValueError('Unknown event-learning option: ' + key)
            setattr(self, key, value)

    @staticmethod
    def _transform_residual_model_features(scaler, features):
        """Apply either a native sklearn scaler or the NumPy-only export."""
        if str(getattr(scaler, 'portable_kind_', '')) == 'standard_scaler_numpy_v1':
            mean = np.asarray(scaler.mean_, dtype=float)
            scale = np.asarray(scaler.scale_, dtype=float)
            if mean.shape != (features.shape[1],) or scale.shape != mean.shape:
                raise ValueError('便携标准化器维度与模型输入不一致')
            return (features - mean) / scale
        return scaler.transform(features)

    @staticmethod
    def _predict_portable_residual_estimator(estimator, features):
        """Run a portable Ridge/HistGBR record without sklearn internals."""
        kind = str(getattr(estimator, 'portable_kind_', ''))
        if kind == 'ridge_numpy_v1':
            return features @ np.asarray(estimator.coef_, dtype=float) + float(estimator.intercept_)
        if kind != 'hist_gradient_boosting_numpy_v1':
            return estimator.predict(features)
        nodes = estimator.nodes_
        tree_offsets = np.asarray(estimator.tree_offsets_, dtype=np.int64)
        node_index = np.broadcast_to(tree_offsets, (features.shape[0], len(tree_offsets))).copy()
        sample_index = np.arange(features.shape[0], dtype=np.int64)[:, None]
        while True:
            is_leaf = nodes['is_leaf'][node_index].astype(bool)
            if np.all(is_leaf):
                break
            active = ~is_leaf
            feature_index = nodes['feature_idx'][node_index].astype(np.int64)
            feature_value = features[sample_index, feature_index]
            go_left = np.where(np.isnan(feature_value), nodes['missing_go_to_left'][node_index].astype(bool), feature_value <= nodes['num_threshold'][node_index])
            next_index = tree_offsets[None, :] + np.where(go_left, nodes['left'][node_index], nodes['right'][node_index]).astype(np.int64)
            node_index[active] = next_index[active]
        return float(estimator.baseline_) + np.sum(nodes['value'][node_index], axis=1)

    def _predict_multioutput_gp(self, model, scaler, features, apply_calibration=True):
        """Return per-axis residual means and calibrated variances.

        Legacy artifacts expose native GP standard deviations.  The axiswise
        hybrid stores blocked-cross-validation RMSE values as conservative,
        constant predictive standard deviations.  Keeping this adapter in one
        place lets all open- and closed-loop safety logic consume either model
        family without changing the 32-feature causal contract.
        """
        if model is None or scaler is None or (not hasattr(model, 'estimators_')):
            raise ValueError('按命令事件 GPR 模型或标准化器未加载')
        features = p2_ablate_features(np.asarray(features, dtype=float).reshape(1, -1),
                                    getattr(model, 'p2_ablation_', 'none'))
        axis_scalers = getattr(model, 'axis_scalers_', None)
        if axis_scalers is not None and len(axis_scalers) != len(model.estimators_):
            raise ValueError('GPR per-axis scaler count does not match estimator count')
        features_scaled = None if axis_scalers is not None else self._transform_residual_model_features(scaler, features)
        mean = np.zeros(4)
        variance = np.zeros(4)
        model_family = str(getattr(self, 'gpr_model_family', getattr(model, 'model_family_', 'gpr'))).strip().lower()
        hybrid_std = np.asarray(getattr(model, 'predictive_std_', np.full(4, np.nan)), dtype=float)
        if model_family == 'axiswise_hybrid_v1' and (hybrid_std.shape != (4,) or np.any(~np.isfinite(hybrid_std)) or np.any(hybrid_std <= 0.0)):
            raise ValueError('混合残差模型缺少四轴有效预测标准差')
        for axis, estimator in enumerate(model.estimators_):
            axis_features = features_scaled if axis_scalers is None else axis_scalers[axis].transform(features)
            if model_family == 'axiswise_hybrid_v1':
                mu_axis = self._predict_portable_residual_estimator(estimator, axis_features)
                mean[axis] = float(mu_axis[0])
                variance[axis] = float(hybrid_std[axis] ** 2)
            else:
                mu_axis, std_axis = estimator.predict(axis_features, return_std=True)
                mean[axis] = float(mu_axis[0])
                variance[axis] = float(std_axis[0] ** 2)
        if hasattr(model, 'physical_mean_coefficients_'):
            mean += p2_physical_mean(features, model.physical_mean_coefficients_, model.play_radii_mm_, model.relaxation_s_, getattr(model, 'p2_ablation_', 'none'))[0]
        if model_family == 'axiswise_hybrid_v1':
            taper_mm = max(float(getattr(self, 'gpr_zero_command_taper_mm', getattr(model, 'zero_command_taper_mm_', 0.05))), 0.0)
            if taper_mm > 0.0:
                candidate_start = 8 if features.shape[1] == 16 else 12
                command_norm = float(np.linalg.norm(features[0, candidate_start:candidate_start + 4]))
                mean *= float(np.clip(command_norm / taper_mm, 0.0, 1.0))
        if apply_calibration:
            scale = np.asarray(getattr(self, 'gpr_std_calibration_scale', np.ones(4)), dtype=float).reshape(4)
            variance *= np.maximum(scale, 1.0) ** 2
        return (mean, variance)

    def _fit_multioutput_gpr(self, features, targets):
        scaler = StandardScaler()
        features = p2_ablate_features(np.asarray(features, dtype=float), self.p2_ablation)
        targets = np.asarray(targets, dtype=float)
        physical = features.shape[1] == P2_FEATURE_COUNT
        coefficients = None
        if physical:
            coefficients = np.zeros((4, 5)) if self.gpr_training_family == 'memory_gpr' else p2_fit_physical_mean(features, targets, self.p2_play_radii_mm, self.p2_relaxation_s, ablation=self.p2_ablation)
            targets = targets - p2_physical_mean(features, coefficients, self.p2_play_radii_mm, self.p2_relaxation_s, self.p2_ablation)
        features_scaled = scaler.fit_transform(features)
        kernel = ConstantKernel(1.0, constant_value_bounds=(0.001, 1000.0)) * Matern(length_scale=np.ones(features_scaled.shape[1]), length_scale_bounds=(0.01, 1000.0), nu=1.5) + WhiteKernel(noise_level=1e-05, noise_level_bounds=(1e-10, 10.0))
        base = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=int(self.gpr_training_n_restarts), normalize_y=True, random_state=int(self.gpr_training_seed), optimizer=gpr_optimizer_global if getattr(self, 'gpr_training_optimize', True) else None)
        model = MultiOutputRegressor(base)
        model.p2_ablation_ = self.p2_ablation
        model.fit(features_scaled, np.asarray(targets, dtype=float))
        if physical:
            model.physical_mean_coefficients_ = coefficients
            model.play_radii_mm_ = np.asarray(self.p2_play_radii_mm, dtype=float)
            model.relaxation_s_ = float(self.p2_relaxation_s)
            model.memory_version_ = P2_MEMORY_VERSION
        for estimator in model.estimators_:
            estimator.optimizer = None
        return (model, scaler)

    @staticmethod
    def _gpr_event_csv_fieldnames():
        return ['SchemaVersion', 'SampleMode', 'SessionId', 'MeasurementPolicy', 'BoundaryIndex', 'SweepPhaseGenerator', 'SweepSeed', 'SweepCarrierPhase1', 'SweepCarrierPhase2', 'SweepCarrierPhase3', 'SweepCarrierPhase4', 'SweepModulationPhase1', 'SweepModulationPhase2', 'SweepModulationPhase3', 'SweepModulationPhase4', 'SweepCandidateCount', 'SweepSelectedCandidate', 'SweepCoverageBinsPerAxis', 'SweepCoverageCells', 'SweepCoverageTotalCells', 'SweepCoverageFraction', 'SweepMaxAbsAxisCorrelation', 'AcquisitionStrategy', 'ActiveSelectionId', 'ActiveScore', 'ActiveGeometryGain', 'ActivePhysicalGain', 'ActiveTaskVariance', 'PlannedSettleMs', 'P2FeatureQueryMonoTime', 'P2MemoryVersion', 'P2PlayRadiusSmallMm', 'P2PlayRadiusLargeMm', 'EventRole', 'SegmentId', 'VisibilityCell', 'RecoveryReason', 'CommandStepCapMm', 'PreviousCommandId', 'MeasurementValid', 'InvalidReason', 'CommandSendWallTime', 'CommandSendMonoTime', 'CommandDoneWallTime', 'CommandDoneMonoTime', 'SettleEndMonoTime', 'BoundaryRecordWallTime', 'BoundaryRecordMonoTime', 'NdiFrameId', 'NdiDeviceTimestamp', 'NdiReceiveWallTime', 'NdiReceiveMonoTime', 'NdiCaptureHostMonoTime', 'NdiFrameIdAtDone', 'ClockSyncValid', 'TrackingQuality', 'NdiAverageFrameCount', 'NdiAverageTargetFrameCount', 'NdiAverageStartFrameId', 'NdiAverageEndFrameId', 'NdiAverageSpanMs', 'NdiAverageTranslationRmsMm', 'NdiAverageRotationRmsDeg', 'NdiAverageQualityMean', 'NdiAverageAxisRmsMaxMm', 'NdiAverageAxisRangeMaxMm', 'MotionMismatchMaxAbsMm', 'MotionMismatchLimitMm', 'X1', 'Y1', 'X2', 'Y2', 'Act_X', 'Act_Y', 'Act_Z', 'Act_Qx', 'Act_Qy', 'Act_Qz', 'Act_Qw', 'Act_Rx', 'Act_Ry', 'Act_Rz', 'Act_q1', 'Act_q2', 'Act_q3', 'Act_q4', 'SpeedSetting', 'AccelerationSetting', 'LoadCondition', 'CalibrationVersion']

    @staticmethod
    def _gpr_sweep_replay_fieldnames():
        """Optional v5 fields; legacy v5 files remain valid without them."""
        return {'MeasurementPolicy', 'SweepPhaseGenerator', 'SweepSeed', 'SweepCarrierPhase1', 'SweepCarrierPhase2', 'SweepCarrierPhase3', 'SweepCarrierPhase4', 'SweepModulationPhase1', 'SweepModulationPhase2', 'SweepModulationPhase3', 'SweepModulationPhase4', 'SweepCandidateCount', 'SweepSelectedCandidate', 'SweepCoverageBinsPerAxis', 'SweepCoverageCells', 'SweepCoverageTotalCells', 'SweepCoverageFraction', 'SweepMaxAbsAxisCorrelation', 'AcquisitionStrategy', 'ActiveSelectionId', 'ActiveScore', 'ActiveGeometryGain', 'ActivePhysicalGain', 'ActiveTaskVariance', 'PlannedSettleMs', 'P2FeatureQueryMonoTime', 'P2MemoryVersion', 'P2PlayRadiusSmallMm', 'P2PlayRadiusLargeMm', 'EventRole', 'SegmentId', 'VisibilityCell', 'RecoveryReason', 'CommandStepCapMm', 'NdiAverageFrameCount', 'NdiAverageTargetFrameCount', 'NdiAverageStartFrameId', 'NdiAverageEndFrameId', 'NdiAverageSpanMs', 'NdiAverageTranslationRmsMm', 'NdiAverageRotationRmsDeg', 'NdiAverageQualityMean', 'NdiAverageAxisRmsMaxMm', 'NdiAverageAxisRangeMaxMm', 'MotionMismatchMaxAbsMm', 'MotionMismatchLimitMm'}

    @classmethod
    def _gpr_event_required_fieldnames(cls):
        return set(cls._gpr_event_csv_fieldnames()).difference(cls._gpr_sweep_replay_fieldnames())

    def _build_gpr_samples_from_csvs(self, csv_paths, purpose='fit'):
        """Validate and merge independent schema-v5 command-event sessions."""
        if purpose not in ('fit', 'evaluation'):
            raise ValueError('Dataset purpose must be fit or evaluation')
        required = self._gpr_event_required_fieldnames()
        samples = []
        source_files = []
        session_ids = set()
        calibration_versions = set()
        condition_signatures = set()
        measurement_policies = set()
        flows, replay_audit = [], []

        def numeric_condition(row, field):
            value = float(row[field])
            if not np.isfinite(value) or value<=0:
                raise ValueError('Speed and acceleration settings must be finite and positive')
            return f'{value:.9g}'
        for path in map(os.path.abspath, csv_paths):
            with open(path, 'r', encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                missing = required.difference(set(reader.fieldnames or []))
            if missing:
                raise ValueError(f'{os.path.basename(path)} 缺少字段: {sorted(missing)}')
            if len(rows) < 4:
                raise ValueError(f'{os.path.basename(path)} 的事件边界少于 4 个')
            if any((int(row['SchemaVersion']) != self.gpr_data_schema_version or row['SampleMode'] != 'command_event_done' for row in rows)):
                raise ValueError(f'{os.path.basename(path)} 不是 schema-v5 命令事件数据')
            file_policies={str(row.get('MeasurementPolicy','legacy_mixed_v0') or 'legacy_mixed_v0').strip() for row in rows}
            if len(file_policies)!=1 or not file_policies.issubset({'raw_single_v1','legacy_mixed_v0'}):
                raise ValueError('A session must use one explicit supported measurement policy')
            measurement_policies.update(file_policies)
            file_sessions = {str(row['SessionId']).strip() for row in rows}
            file_calibrations = {str(row['CalibrationVersion']).strip() for row in rows}
            if '' in file_sessions or len(file_sessions) != 1:
                raise ValueError('每个 CSV 必须且只能包含一个非空 SessionId')
            if '' in file_calibrations or len(file_calibrations) != 1:
                raise ValueError('每个 CSV 必须且只能包含一个标定版本')
            session_id = next(iter(file_sessions))
            if session_id in session_ids:
                raise ValueError(f'重复选择了 SessionId={session_id}')
            file_conditions = {(numeric_condition(row, 'SpeedSetting'), numeric_condition(row, 'AccelerationSetting'), str(row['LoadCondition']).strip()) for row in rows}
            if len(file_conditions) != 1:
                raise ValueError('一个会话内的速度、加速度和头模负载条件必须固定')
            if next(iter(file_conditions))[2].lower() != 'head_phantom':
                raise ValueError(f'{os.path.basename(path)} 不是 head_phantom 条件数据')
            boundary = np.asarray([int(row['BoundaryIndex']) for row in rows])
            previous = np.asarray([int(row['PreviousCommandId']) for row in rows])
            expected_previous = boundary - 1
            expected_previous[0] = -1
            if boundary[0] != 0 or np.any(np.diff(boundary) != 1):
                raise ValueError('BoundaryIndex 必须从 0 开始连续递增')
            if not np.array_equal(previous, expected_previous):
                raise ValueError('PreviousCommandId 与事件边界不一致')
            validity = np.asarray([int(row['MeasurementValid']) for row in rows])
            if np.any((validity != 0) & (validity != 1)):
                raise ValueError('MeasurementValid 必须为 0 或 1')
            row_valid = validity.astype(bool)
            invalid_reasons = np.asarray([str(row['InvalidReason']).strip() for row in rows], dtype=object)
            invalid_rows = np.flatnonzero(~row_valid)
            if np.any(~row_valid & (invalid_reasons == '')):
                raise ValueError('无效事件边界必须记录 InvalidReason')
            frame_id = np.asarray([int(row['NdiFrameId']) for row in rows])
            device_time = np.asarray([float(row['NdiDeviceTimestamp']) for row in rows])
            receive_time = np.asarray([float(row['NdiReceiveMonoTime']) for row in rows])
            record_time = np.asarray([float(row['BoundaryRecordMonoTime']) for row in rows])
            send_time = np.asarray([float(row['CommandSendMonoTime']) for row in rows])
            done_time = np.asarray([float(row['CommandDoneMonoTime']) for row in rows])
            settle_time = np.asarray([float(row['SettleEndMonoTime']) for row in rows])
            frame_at_done = np.asarray([int(row['NdiFrameIdAtDone']) for row in rows])
            if np.any(~np.isfinite(record_time)) or np.any(np.diff(record_time) <= 0):
                raise ValueError('事件记录时间必须有限且严格递增')
            if np.isfinite(send_time[0]) or np.isfinite(done_time[0]) or np.isfinite(settle_time[0]) or (frame_at_done[0] >= 0):
                raise ValueError('初始边界不能引用之前的命令')
            valid_index = np.flatnonzero(row_valid)
            if np.any(row_valid & ((frame_id < 0) | ~np.isfinite(device_time) | ~np.isfinite(receive_time))):
                raise ValueError('有效边界缺少 NDI 帧号或时间戳')
            if np.any(row_valid & ((record_time - receive_time < -0.001) | (record_time - receive_time > self.gpr_max_ndi_frame_age_s))):
                raise ValueError('有效 NDI 帧超过新鲜度阈值')
            if len(valid_index) > 1 and (np.any(np.diff(frame_id[valid_index]) <= 0) or np.any(np.diff(device_time[valid_index]) < 0) or np.any(np.diff(receive_time[valid_index]) <= 0)):
                raise ValueError('有效 NDI 帧号和时间戳必须严格递增')
            if np.any(~np.isfinite(send_time[1:])):
                raise ValueError('非初始边界缺少命令发送时间')
            if np.any(row_valid[1:] & ~np.isfinite(done_time[1:])):
                raise ValueError('有效边界缺少 GOTO DONE 时间')
            complete = np.isfinite(done_time[1:])
            incomplete_rows = np.flatnonzero(~complete) + 1
            if len(incomplete_rows) and (len(incomplete_rows) != 1 or incomplete_rows[0] != len(rows) - 1 or row_valid[incomplete_rows[0]] or (invalid_reasons[incomplete_rows[0]] != 'command_completion_timeout')):
                raise ValueError('只有终止行的命令超时可以缺少 GOTO DONE')
            if np.any(complete & ((done_time[1:] <= send_time[1:]) | ~np.isfinite(settle_time[1:]) | (settle_time[1:] < done_time[1:]) | (record_time[1:] < done_time[1:]) | (done_time[1:] - send_time[1:] > self.gpr_command_timeout_s))):
                raise ValueError('命令发送、完成、停稳和记录时间顺序无效')
            if np.any(row_valid[1:] & ((receive_time[1:] < settle_time[1:]) | (frame_id[1:] <= frame_at_done[1:]))):
                raise ValueError('有效 NDI 量测不是 GOTO DONE 停稳后的新帧')
            q_m = np.asarray([[float(row[f'Act_q{axis + 1}']) for axis in range(4)] for row in rows])
            q_c = np.asarray([[float(row[name]) for name in ('X1', 'Y1', 'X2', 'Y2')] for row in rows])
            row_valid &= np.all(np.isfinite(q_m), axis=1)
            row_valid &= np.all(np.isfinite(q_c), axis=1)
            event_role = np.asarray([str(row.get('EventRole', 'sweep') or 'sweep').strip().lower() for row in rows], dtype=object)
            try:
                segment_id = np.asarray([int(row.get('SegmentId', 0) or 0) for row in rows], dtype=int)
            except ValueError as exc:
                raise ValueError('SegmentId must be an integer') from exc
            trainable_row = row_valid & (event_role == 'sweep')
            guarded_row_valid = trainable_row.copy()
            edge_guard = max(int(self.gpr_ndi_edge_guard_boundaries), 0)
            if edge_guard > 0:
                for invalid_index in np.flatnonzero(~trainable_row):
                    start = max(0, int(invalid_index) - edge_guard)
                    stop = min(len(guarded_row_valid), int(invalid_index) + edge_guard + 1)
                    guarded_row_valid[start:stop] = False
            command = q_c[1:] - q_c[:-1]
            realized = q_m[1:] - q_m[:-1]
            residual = command - realized
            command_valid = np.all(np.isfinite(command), axis=1) & (np.max(np.abs(command), axis=1) <= self.gpr_training_max_increment)
            motion_mismatch_valid = np.all(np.isfinite(residual), axis=1) & (np.max(np.abs(residual), axis=1) <= float(self.gpr_ndi_max_boundary_jump_mm))
            memory_features = {}
            feature_version = int(getattr(self, 'gpr_sample_feature_version', self.gpr_model_format_version))
            if feature_version == 10:
                logged_versions = {str(row.get('P2MemoryVersion', '')).strip() for row in rows}
                if len(logged_versions) != 1 or not logged_versions.issubset(P2_REPLAYABLE_VERSIONS):
                    raise ValueError('CSV physical-memory state version mismatch')
                if logged_versions != {''}:
                    logged_radii = np.asarray([[float(row.get('P2PlayRadiusSmallMm', np.nan)), float(row.get('P2PlayRadiusLargeMm', np.nan))] for row in rows], dtype=float)
                    if np.any(~np.isfinite(logged_radii)) or np.any(logged_radii <= 0) or (not np.all(logged_radii == logged_radii[0])):
                        raise ValueError('invalid or changing CSV physical-memory radii')
                memory = PhysicalEventMemory(self.p2_play_radii_mm)
                for k in range(len(rows)):
                    current_ok = bool(row_valid[k] and event_role[k] == 'sweep')
                    # Model-domain/label filters must never change the physical trajectory.
                    previous_ok = bool(k > 0 and np.isfinite(command[k - 1]).all() and (event_role[k - 1] == 'sweep') and (segment_id[k] == segment_id[k - 1]))
                    if current_ok and previous_ok:
                        memory.commit(command[k - 1], send_time[k], done_time[k], record_time[k])
                    else:
                        memory.reset(record_time[k])
                    if k >= 2 and k + 1 < len(rows) and np.all(row_valid[k - 2:k + 2]) and (np.unique(segment_id[k - 2:k + 2]).size == 1) and np.all(event_role[k - 2:k + 2] == 'sweep') and np.isfinite(send_time[k + 1]):
                        planned = float(rows[k + 1].get('PlannedSettleMs', self.gpr_sweep_settle_ms) or self.gpr_sweep_settle_ms) / 1000.0
                        query = float(rows[k + 1].get('P2FeatureQueryMonoTime', send_time[k + 1]) or send_time[k + 1])
                        if not np.isfinite(query) or query > send_time[k + 1] + 1e-06:
                            raise ValueError('physical-memory query time must precede command send')
                        memory_features[k] = p2_causal_features([q_m[k], q_m[k - 1], q_m[k - 2]], [command[k - 1], command[k - 2]], q_c[k], command[k], memory, query, planned, float(rows[k]['SpeedSetting']), float(rows[k]['AccelerationSetting']))
                        replay_audit.append(dict(session_id=session_id, sample_k=k,
                            commits=memory.commits, features=memory_features[k].tolist()))
            flow = dict(session_id=session_id, source_file=path, boundaries=len(rows),
                        invalid_boundary_indices=np.flatnonzero(~trainable_row).tolist(),
                        candidate_windows=max(0,len(rows)-3), physical_valid_windows=0,
                        excluded_physical=0, excluded_fit_domain=0, excluded_guard=0,
                        included=0, intrinsic_fit_indices=[], purpose=purpose)
            for index in range(2, len(rows) - 1):
                boundary_slice = slice(index - 2, index + 2)
                physical_ok = (np.all(trainable_row[boundary_slice]) and
                               np.unique(segment_id[boundary_slice]).size == 1 and
                               np.isfinite(residual[index]).all())
                if not physical_ok:
                    flow['excluded_physical'] += 1
                    continue
                flow['physical_valid_windows'] += 1
                domain = bool(np.all(command_valid[index-2:index+1]) and
                              np.all(motion_mismatch_valid[index-2:index+1]))
                guard_ok = bool(np.all(guarded_row_valid[boundary_slice]))
                if domain: flow['intrinsic_fit_indices'].append(index)
                if purpose == 'fit' and not domain:
                    flow['excluded_fit_domain'] += 1
                    continue
                if purpose == 'fit' and not guard_ok:
                    flow['excluded_guard'] += 1
                    continue
                reversal = ((np.abs(command[index]) > self.gpr_reversal_command_epsilon) & (np.abs(command[index - 1]) > self.gpr_reversal_command_epsilon) & (command[index] * command[index - 1] < 0.0)).astype(float)
                samples.append({'features': np.concatenate([q_m[index], q_m[index - 1], command[index], command[index - 1]]) if feature_version == 8 else memory_features[index] if feature_version == 10 else np.concatenate([q_m[index], q_m[index - 1], q_m[index - 2], command[index], command[index - 1], command[index - 2], q_c[index] - q_m[index], reversal]), 'target_delta_d': residual[index], 'du_k': command[index], 'du_prev': command[index - 1], 'session_id': session_id, 'source_file': os.path.basename(path), 'sample_k': int(boundary[index]),
                    'fit_domain':int(domain), 'fit_guard':int(guard_ok),
                    'current_command_in_domain':int(command_valid[index]),
                    'current_label_in_domain':int(motion_mismatch_valid[index]),
                    'planned_settle_s':float(rows[index+1].get('PlannedSettleMs',self.gpr_sweep_settle_ms) or self.gpr_sweep_settle_ms)/1000.})
                flow['included'] += 1
            flows.append(flow)
            session_ids.add(session_id)
            calibration_versions.update(file_calibrations)
            condition_signatures.update(file_conditions)
            source_files.append(path)
        if len(measurement_policies)!=1:
            raise ValueError('Cannot merge different measurement policies')
        if len(calibration_versions) != 1:
            raise ValueError('不能合并不同标定版本的会话')
        if len(condition_signatures) != 1:
            raise ValueError('不能合并不同速度、加速度或负载条件的会话')
        return (samples, {'session_ids': sorted(session_ids), 'source_files': source_files, 'calibration_version': next(iter(calibration_versions)), 'condition_signature': next(iter(condition_signatures)), 'measurement_policy': next(iter(measurement_policies)), 'purpose':purpose, 'sample_flow':flows, 'memory_replay':replay_audit})
