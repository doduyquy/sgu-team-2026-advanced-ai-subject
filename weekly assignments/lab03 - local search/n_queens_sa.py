"""Simulated Annealing dùng chung cho Bài 4 (Task 5) và Bài 5."""

import math

import numpy as np


def conflicts(board):
    """Đếm cặp hậu xung đột, cùng cách biểu diễn với n_queens.ipynb."""
    n = len(board)
    rows = [0] * n
    diagonals_down = [0] * (2 * n)
    diagonals_up = [0] * (2 * n)
    for column, row in enumerate(board):
        rows[row] += 1
        diagonals_down[column + row] += 1
        diagonals_up[column - row + n] += 1
    return sum(count * (count - 1) // 2
               for count in rows + diagonals_down + diagonals_up)


def simulated_annealing(initial_board, schedule, max_steps=5000, seed=0):
    """Di chuyển một hậu sang hàng khác trong cột; lưu nghiệm tốt nhất.

    Dừng khi tìm được 0 xung đột, hết ngân sách hoặc T <= 0. Lịch nhiệt
    được gọi từ t=1 nên có thể dùng trực tiếp công thức logarithmic.
    """
    board = np.asarray(initial_board, dtype=int).copy()
    n = len(board)
    if board.ndim != 1 or n < 1 or np.any((board < 0) | (board >= n)):
        raise ValueError('Bàn cờ phải có n hàng hợp lệ trong khoảng 0..n-1.')
    if max_steps < 1:
        raise ValueError('max_steps phải dương.')

    rng = np.random.default_rng(seed)
    current_f = conflicts(board)
    best_board, best_f = board.copy(), current_f
    current_history, best_history = [current_f], [best_f]
    worse_proposed = worse_accepted = steps = 0
    stop_reason = 'budget'

    for t in range(1, max_steps + 1):
        if best_f == 0:
            stop_reason = 'solved'
            break
        temperature = float(schedule(t))
        if not math.isfinite(temperature):
            raise ValueError('Lịch nhiệt phải trả về nhiệt độ hữu hạn.')
        if temperature <= 0:
            stop_reason = 'cooled'
            break

        column = int(rng.integers(n))
        new_row = int(rng.integers(n - 1))
        if new_row >= board[column]:
            new_row += 1
        candidate = board.copy()
        candidate[column] = new_row
        candidate_f = conflicts(candidate)
        delta = candidate_f - current_f
        # Rút một số mỗi bước, kể cả khi delta <= 0, để giữ nhịp RNG nhất quán.
        draw = rng.random()
        accepted = delta <= 0 or draw < math.exp(-delta / temperature)
        if delta > 0:
            worse_proposed += 1
            worse_accepted += int(accepted)
        if accepted:
            board, current_f = candidate, candidate_f
            if current_f < best_f:
                best_board, best_f = board.copy(), current_f
        current_history.append(current_f)
        best_history.append(best_f)
        steps = t

    if best_f == 0:
        stop_reason = 'solved'
    return {
        'best_board': best_board,
        'best_f': best_f,
        'final_f': current_f,
        'steps': steps,
        'stop_reason': stop_reason,
        'current_history': np.asarray(current_history),
        'best_history': np.asarray(best_history),
        'worse_proposed': worse_proposed,
        'worse_accepted': worse_accepted,
    }
