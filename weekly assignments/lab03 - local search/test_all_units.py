import unittest
import os
import sys
import math
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from numba import njit

# Đảm bảo đường dẫn thực thi chuẩn
LAB_DIR = Path(__file__).parent.resolve()
os.chdir(LAB_DIR)
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import maze_helper as mh
import n_queens_sa as sa_mod

# --- CÁC HÀM CỐT LÕI CỦA TRI (BÀI 2, 3, 4, 8) ---
def conflicts(board):
    n = len(board)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                total += 1
    return total

def check_board(board):
    q = np.asarray(board)
    if q.ndim != 1 or len(q) < 2 or q.dtype.kind not in 'iu':
        raise ValueError('board phải là vector số nguyên, n >= 2.')
    n = len(q)
    if np.any((q < 0) | (q >= n)):
        raise ValueError('Mỗi phần tử phải thuộc [0, n-1].')
    return np.array(q, dtype=int, copy=True)

def local_neighbors(board, operator='any'):
    q = check_board(board)
    n = len(q)
    for col in range(n):
        if operator == 'adjacent':
            candidate_rows = [row for row in (q[col] - 1, q[col] + 1) if 0 <= row < n]
        else:
            candidate_rows = [row for row in range(n) if row != q[col]]
        for row in candidate_rows:
            candidate = q.copy()
            candidate[col] = row
            yield candidate

def random_neighbor(board, rng, operator='any'):
    q = np.array(board, copy=True)
    n = len(q)
    col = int(rng.integers(n))
    if operator == 'adjacent':
        moves = [r for r in (q[col] - 1, q[col] + 1) if 0 <= r < n]
        if not moves:
            return q
        row = moves[int(rng.integers(len(moves)))]
    else:
        row = int(rng.integers(n - 1))
        if row >= q[col]:
            row += 1
    q[col] = row
    return q

def first_choice_trial_limit(n, miss_probability=0.05, operator='any'):
    if operator == 'adjacent':
        neighbors_count = max(2, 2 * n - 2)
    else:
        neighbors_count = n * (n - 1)
    return math.ceil(math.log(miss_probability) / math.log1p(-1 / neighbors_count))

def result_dict(method, q, f, history, candidate_evals, reason, attempts=None):
    h_list = [int(x) for x in history]
    return {
        'algorithm': method,
        'board': q.copy(),
        'conflicts': int(f),
        'best_board': q.copy(),
        'best_f': int(f),
        'history': tuple(h_list),
        'current_history': h_list,
        'best_history': h_list,
        'iterations': len(history) - 1,
        'steps': len(history) - 1,
        'candidate_evaluations': int(candidate_evals),
        'objective_evaluations': int(candidate_evals + 1),
        'evaluations': int(candidate_evals + 1),
        'attempts': int(candidate_evals if attempts is None else attempts),
        'success': bool(f == 0),
        'stop_reason': reason,
        'reason': 'solved' if f == 0 else reason,
    }

def hill_climb_steepest(board, seed=None, max_steps=100, max_evaluations=5000, operator='any'):
    q = check_board(board)
    n = len(q)
    f = int(conflicts(q))
    history = [f]
    evaluations = 0
    reason = 'step_budget'
    while len(history) - 1 < max_steps:
        if f == 0:
            reason = 'solved'
            break
        m = 2 * n if operator == 'adjacent' else n * (n - 1)
        if evaluations + m > max_evaluations:
            reason = 'evaluation_budget'
            break
        best = None
        best_f = f
        for candidate in local_neighbors(q, operator=operator):
            fc = int(conflicts(candidate))
            evaluations += 1
            if fc < best_f:
                best, best_f = candidate, fc
        if best is None:
            reason = 'local_optimum'
            break
        q, f = best, best_f
        history.append(f)
    if f == 0:
        reason = 'solved'
    return result_dict('Steepest-ascent', q, f, history, evaluations, reason)

def hill_climb_stochastic_all(board, seed=0, max_steps=100, max_evaluations=5000, operator='any'):
    q = check_board(board)
    rng = np.random.default_rng(seed)
    n = len(q)
    f = int(conflicts(q))
    history = [f]
    evaluations = 0
    reason = 'step_budget'
    while len(history) - 1 < max_steps:
        if f == 0:
            reason = 'solved'
            break
        m = 2 * n if operator == 'adjacent' else n * (n - 1)
        if evaluations + m > max_evaluations:
            reason = 'evaluation_budget'
            break
        improvements = []
        for candidate in local_neighbors(q, operator=operator):
            fc = int(conflicts(candidate))
            evaluations += 1
            if fc < f:
                improvements.append((candidate, fc))
        if not improvements:
            reason = 'local_optimum'
            break
        index = int(rng.integers(len(improvements)))
        q, f = improvements[index]
        history.append(f)
    if f == 0:
        reason = 'solved'
    return result_dict('Stochastic-all', q, f, history, evaluations, reason)

def hill_climb_first_choice(board, seed=0, max_steps=100,
                            max_evaluations=5000, max_no_improvement_trials=None,
                            operator='any'):
    q = check_board(board)
    n = len(q)
    if max_no_improvement_trials is None:
        max_no_improvement_trials = first_choice_trial_limit(n, operator=operator)
    rng = np.random.default_rng(seed)
    f = int(conflicts(q))
    history = [f]
    evaluations = 0
    unsuccessful_streak = 0
    reason = 'step_budget'
    while len(history) - 1 < max_steps:
        if f == 0:
            reason = 'solved'
            break
        if evaluations >= max_evaluations:
            reason = 'evaluation_budget'
            break
        if unsuccessful_streak >= max_no_improvement_trials:
            reason = 'trial_limit'
            break
        candidate = random_neighbor(q, rng, operator=operator)
        fc = int(conflicts(candidate))
        evaluations += 1
        if fc < f:
            q, f = candidate, fc
            history.append(f)
            unsuccessful_streak = 0
        else:
            unsuccessful_streak += 1
    if f == 0:
        reason = 'solved'
    return result_dict('First-choice', q, f, history, evaluations, reason)

def temperature_exponential(t, T0, alpha):
    return float(T0 * alpha**t)

def simulated_annealing(initial_board, schedule, seed=0, max_steps=2999,
                        total_evaluation_budget=3000):
    board = check_board(initial_board)
    rng = np.random.default_rng(seed)
    current_f = int(conflicts(board))
    initial_f = current_f
    best_board, best_f = board.copy(), current_f
    current_history, best_history = [current_f], [best_f]
    worse_proposed = worse_accepted = plateau_accepted = 0
    steps = 0
    reason = 'step_budget'
    for t in range(1, max_steps + 1):
        if best_f == 0:
            reason = 'solved'
            break
        if steps + 1 >= total_evaluation_budget:
            reason = 'evaluation_budget'
            break
        temperature = float(schedule(t))
        if not math.isfinite(temperature):
            raise ValueError('T(t) không được vô hạn hoặc NaN.')
        if temperature <= 0:
            reason = 'cooled'
            break
        candidate = random_neighbor(board, rng)
        candidate_f = int(conflicts(candidate))
        delta = candidate_f - current_f
        if delta > 0:
            worse_proposed += 1
            accepted = math.log(max(float(rng.random()), np.finfo(float).tiny)) < -delta / temperature
            worse_accepted += int(accepted)
        else:
            accepted = True
            plateau_accepted += int(delta == 0)
        if accepted:
            board, current_f = candidate, candidate_f
            if current_f < best_f:
                best_board, best_f = board.copy(), current_f
        steps += 1
        current_history.append(int(current_f))
        best_history.append(int(best_f))
    if best_f == 0:
        reason = 'solved'
    return {
        'algorithm': 'Simulated Annealing',
        'board': best_board.copy(), 'conflicts': int(best_f),
        'best_board': best_board.copy(), 'best_f': int(best_f),
        'current_board': board.copy(), 'final_f': int(current_f),
        'initial_f': int(initial_f), 'success': best_f == 0,
        'steps': steps, 'iterations': steps,
        'candidate_evaluations': steps, 'objective_evaluations': steps + 1,
        'evaluations': steps + 1,
        'stop_reason': reason, 'reason': reason,
        'current_history': current_history,
        'best_history': best_history,
        'history': tuple(best_history),
        'worse_proposed': worse_proposed, 'worse_accepted': worse_accepted,
        'plateau_accepted': plateau_accepted,
    }


# ================= UNITTEST TESTCASES =================

class TestBai1Maze(unittest.TestCase):
    """Bài 1: Mê cung và Hill Climbing"""
    def setUp(self):
        self.maze_file = LAB_DIR / "L_maze.txt"
        with open(self.maze_file, 'r', encoding='utf-8') as f:
            self.maze = mh.parse_maze(f.read())
        self.start = mh.find_pos(self.maze, 'S')
        self.goal = mh.find_pos(self.maze, 'G')

    def test_file_exists_and_parsed(self):
        self.assertTrue(self.maze_file.is_file())
        self.assertEqual(self.start, (9, 5))
        self.assertEqual(self.goal, (2, 12))

    def test_manhattan_distance(self):
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        self.assertEqual(manhattan(self.start, self.goal), 14)
        self.assertEqual(manhattan((0, 0), (3, 4)), 7)

    def test_hill_climbing_termination(self):
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        directions = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}
        order = ['N', 'E', 'S', 'W']
        curr = self.start
        steps = 0
        while curr != self.goal and steps < 200:
            best_nbr = None
            best_dist = manhattan(curr, self.goal)
            for d in order:
                dr, dc = directions[d]
                nbr = (curr[0] + dr, curr[1] + dc)
                if 0 <= nbr[0] < len(self.maze) and 0 <= nbr[1] < len(self.maze[0]) and self.maze[nbr[0]][nbr[1]] != 'X':
                    dist = manhattan(nbr, self.goal)
                    if dist < best_dist:
                        best_nbr = nbr
                        best_dist = dist
                        break
            if best_nbr is None:
                break
            curr = best_nbr
            steps += 1
        self.assertEqual(curr, (5, 9))
        self.assertEqual(steps, 8)


class TestBai2NQueens(unittest.TestCase):
    """Bài 2: Mô hình hóa n-Queens và hàm xung đột"""
    def test_conflict_counter(self):
        self.assertEqual(conflicts([1, 3, 0, 2]), 0)
        self.assertEqual(conflicts([0, 4, 7, 5, 2, 6, 1, 3]), 0)
        self.assertEqual(conflicts([0, 0, 0, 0]), 6)
        self.assertEqual(conflicts([0, 1, 2, 3]), 6)

    def test_board_validation(self):
        self.assertTrue(np.array_equal(check_board([1, 3, 0, 2]), np.array([1, 3, 0, 2])))
        with self.assertRaises(ValueError):
            check_board([4, 0, 0, 0])
        with self.assertRaises(ValueError):
            check_board([-1, 0, 0, 0])
        with self.assertRaises(ValueError):
            check_board([0])

    def test_neighbor_count(self):
        nbrs4 = list(local_neighbors([1, 3, 0, 2], operator='any'))
        self.assertEqual(len(nbrs4), 4 * 3)
        nbrs8 = list(local_neighbors([0, 4, 7, 5, 2, 6, 1, 3], operator='any'))
        self.assertEqual(len(nbrs8), 8 * 7)
        nbrs_adj = list(local_neighbors([1, 3, 0, 2], operator='adjacent'))
        self.assertTrue(4 <= len(nbrs_adj) <= 8)


class TestBai3HillClimbing(unittest.TestCase):
    """Bài 3: Ba biến thể Hill Climbing"""
    def test_monotonic_improvement(self):
        test_board = np.array([0, 0, 0, 0])
        for name, fn in [
            ('Steepest', hill_climb_steepest),
            ('Stochastic', hill_climb_stochastic_all),
            ('First-choice', hill_climb_first_choice)
        ]:
            res = fn(test_board, max_evaluations=5000)
            self.assertTrue(np.all(np.diff(res['best_history']) <= 0), f"{name} vi phạm tính đơn điệu không tăng")
            self.assertEqual(res['best_f'], conflicts(res['best_board']))
            self.assertLessEqual(res['evaluations'], 5000)

    def test_input_board_immutability(self):
        board = np.array([3, 2, 1, 0])
        copy_board = board.copy()
        _ = hill_climb_steepest(board, max_evaluations=50)
        self.assertTrue(np.array_equal(board, copy_board))


class TestBai4RandomRestartAndSA(unittest.TestCase):
    """Bài 4: Random Restart và Simulated Annealing"""
    def test_sa_worse_move_acceptance(self):
        schedule = lambda t: temperature_exponential(t, 5.0, 0.995)
        res = simulated_annealing(np.zeros(8, dtype=int), schedule, seed=42, total_evaluation_budget=1000)
        self.assertLessEqual(res['evaluations'], 1000)
        self.assertGreater(res['worse_proposed'], 0)
        self.assertGreater(res['worse_accepted'], 0)
        self.assertEqual(res['best_f'], conflicts(res['best_board']))
        self.assertLessEqual(res['best_f'], min(res['current_history']))

    def test_sa_reproducibility(self):
        schedule = lambda t: temperature_exponential(t, 5.0, 0.995)
        board = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        r1 = simulated_annealing(board, schedule, seed=123, max_steps=200, total_evaluation_budget=201)
        r2 = simulated_annealing(board, schedule, seed=123, max_steps=200, total_evaluation_budget=201)
        self.assertTrue(np.array_equal(r1['best_board'], r2['best_board']))
        self.assertEqual(r1['current_history'], r2['current_history'])


class TestBai5Schedules(unittest.TestCase):
    """Bài 5: Bốn lịch làm nguội SA và module n_queens_sa"""
    def test_module_interfaces(self):
        self.assertTrue(hasattr(sa_mod, 'simulated_annealing'))
        self.assertTrue(hasattr(sa_mod, 'conflicts'))

    def test_cooling_schedules(self):
        t = 10
        T0 = 5.0
        self.assertAlmostEqual(temperature_exponential(t, T0, 0.95), T0 * (0.95**10))
        linear_t = T0 - 0.05 * t
        self.assertAlmostEqual(linear_t, 4.5)


class TestBai6NumbaConflicts(unittest.TestCase):
    """Bài 6: Đếm xung đột tối ưu với Numba JIT"""
    def setUp(self):
        def comb2(n): return n * (n - 1) // 2
        self.comb2 = comb2

    def test_consistency_across_three_implementations(self):
        def conflicts_naive(board):
            n = len(board)
            total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                        total += 1
            return total

        def conflicts_fast(board):
            n = len(board)
            rows = np.zeros(n, dtype=np.int64)
            d1 = np.zeros(2 * n - 1, dtype=np.int64)
            d2 = np.zeros(2 * n - 1, dtype=np.int64)
            for col in range(n):
                r = board[col]
                rows[r] += 1
                d1[col - r + n - 1] += 1
                d2[col + r] += 1
            total = 0
            for c in rows: total += self.comb2(c)
            for c in d1: total += self.comb2(c)
            for c in d2: total += self.comb2(c)
            return total

        @njit
        def conflicts_numba(board):
            n = len(board)
            rows = np.zeros(n, dtype=np.int64)
            d1 = np.zeros(2 * n - 1, dtype=np.int64)
            d2 = np.zeros(2 * n - 1, dtype=np.int64)
            for col in range(n):
                r = board[col]
                rows[r] += 1
                d1[col - r + n - 1] += 1
                d2[col + r] += 1
            total = 0
            for i in range(n):
                c = rows[i]
                total += c * (c - 1) // 2
            for i in range(2 * n - 1):
                c = d1[i]
                total += c * (c - 1) // 2
                c2 = d2[i]
                total += c2 * (c2 - 1) // 2
            return total

        # Warm-up
        _ = conflicts_numba(np.array([0, 1, 2, 3], dtype=np.int64))

        rng = np.random.default_rng(2026)
        for size in [4, 8, 16, 32]:
            for _ in range(5):
                b = rng.integers(0, size, size=size).astype(np.int64)
                c1 = conflicts_naive(b)
                c2 = conflicts_fast(b)
                c3 = conflicts_numba(b)
                self.assertEqual(c1, c2)
                self.assertEqual(c2, c3)


class TestBai7TSP(unittest.TestCase):
    """Bài 7: Traveling Salesman Problem (TSP)"""
    def test_tour_is_valid_closed_permutation(self):
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        tour = np.array([0, 1, 2, 3])
        self.assertTrue(np.array_equal(np.sort(tour), np.arange(4)))

        def tour_length(tsp, tour):
            n = len(tour)
            dists = squareform(pdist(tsp))
            length = 0.0
            for i in range(n):
                length += dists[tour[i], tour[(i + 1) % n]]
            return length

        self.assertAlmostEqual(tour_length(points, tour), 4.0)

    def test_swap_tour_operator(self):
        def swap_tour(tour, i, j):
            new_t = tour.copy()
            new_t[i], new_t[j] = new_t[j], new_t[i]
            return new_t
        t = np.array([0, 1, 2, 3])
        swapped = swap_tour(t, 1, 2)
        self.assertTrue(np.array_equal(swapped, np.array([0, 2, 1, 3])))


class TestBai8Task6Integration(unittest.TestCase):
    """Bài 8: Tích hợp Task 6 với các thuật toán của Tri"""
    def test_budget_compliance_and_oracle(self):
        def queens_steepest(board, max_evals=5000, seed=0):
            return hill_climb_steepest(board, seed=seed, max_steps=max_evals, max_evaluations=max_evals)
        def queens_stochastic(board, max_evals=5000, seed=0):
            return hill_climb_stochastic_all(board, seed=seed, max_steps=max_evals, max_evaluations=max_evals)
        def queens_first_choice(board, max_evals=5000, seed=0):
            return hill_climb_first_choice(board, seed=seed, max_steps=max_evals, max_evaluations=max_evals)
        def queens_sa(board, max_evals=5000, seed=0):
            return simulated_annealing(board, schedule=lambda t: temperature_exponential(t, 5.0, 0.995),
                                       seed=seed, max_steps=max_evals - 1, total_evaluation_budget=max_evals)

        algs = {
            'Steepest': queens_steepest,
            'Stochastic': queens_stochastic,
            'First-choice': queens_first_choice,
            'SA': queens_sa
        }
        for n in [4, 8]:
            for trial in range(5):
                seed = 80000 + 1000 * n + trial
                board = np.random.default_rng(seed).integers(0, n, size=n)
                untouched = board.copy()
                for name, fn in algs.items():
                    res = fn(board, max_evals=5000, seed=seed + 10000)
                    self.assertLessEqual(res['evaluations'], 5000, f"{name} vượt 5000 đánh giá")
                    self.assertEqual(res['best_f'], conflicts(res['best_board']), f"{name} best_f không khớp")
                    self.assertTrue(np.array_equal(board, untouched), f"{name} làm thay đổi input")
                    self.assertTrue(np.all(np.diff(res['best_history']) <= 0), f"{name} best_history không giảm")


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    print("\n" + "="*50)
    print(f"TOTAL TESTS:   {result.testsRun}")
    print(f"PASSED:        {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"FAILED:        {len(result.failures)}")
    print(f"ERRORS:        {len(result.errors)}")
    print(f"SKIPPED:       {len(result.skipped)}")
    print("="*50)
    sys.exit(0 if result.wasSuccessful() else 1)
