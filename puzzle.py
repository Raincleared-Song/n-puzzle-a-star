import heapq
import random
import numpy as np
from enum import Enum
from typing import List, Optional, Set, Dict, Tuple


class StepDirection(Enum):
    ROOT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class PuzzleNode:
    def __init__(self, n, status: Optional[List[int]] = None, log: bool = False,
                 parent=None, current_cost: int = -1, estimate_cost: int = -1, zero_pos: int = -1):
        self.n = n
        self.total_n = n * n
        self.parent: Optional[PuzzleNode] = parent
        # self.last_step: StepDirection = last_step
        # 0 代表空位
        if status is None:
            # random initialization
            self.status = list(range(self.total_n))
            random.shuffle(self.status)
        else:
            assert len(status) == self.total_n
            self.status = status
        # g(n), can be calculated efficiently from the parent node
        self.current_cost = 0 if current_cost == -1 else current_cost
        # h(n), can be calculated efficiently from the parent node
        self.estimate_cost = self.cal_total_estimate_cost() if estimate_cost == -1 else estimate_cost
        if log:
            self.print_status()
            print(f"Current Cost: {self.current_cost}; Estimate Cost: {self.estimate_cost};"
                  f" Total Cost: {self.total_cost}")
        self.zero_pos = self.status.index(0) if zero_pos == -1 else zero_pos

    @property
    def total_cost(self) -> int:  # f(n)
        return self.current_cost + self.estimate_cost

    def print_status(self):
        print('Current Status:')
        for i in range(self.n):
            for j in range(self.n):
                pos = self.n * i + j
                print(f'{self.status[pos]: 3d}', end=' ')
            print()

    def cal_total_estimate_cost(self):
        """calculate h(n), h(target) = 0"""
        total_distance = 0
        for c_pos, item in enumerate(self.status):
            if item == 0:
                continue
            t_pos = item - 1
            cx, cy = c_pos // self.n, c_pos % self.n
            tx, ty = t_pos // self.n, t_pos % self.n
            total_distance += abs(cx - tx) + abs(cy - ty)
        return total_distance

    def cal_single_estimate_cost(self, c_pos: int):
        """calculate h(n) for a single element"""
        assert c_pos != self.zero_pos
        cx, cy = c_pos // self.n, c_pos % self.n
        t_pos = self.status[c_pos] - 1
        tx, ty = t_pos // self.n, t_pos % self.n
        return abs(cx - tx) + abs(cy - ty)

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def __le__(self, other):
        return self.total_cost <= other.total_cost

    @staticmethod
    def get_fingerprint(status: List[int]) -> str:
        return ",".join([str(x) for x in status])

    def check_solvable(self):
        """
            check if a puzzle is solvable by the number of inversions
            if n is odd: puzzle is solvable if and only if
                the number of inversions is even.
            if n is even: puzzle is solvable if and only if
                the row index (start from 0) + the number of inversions is odd.
        """
        num_inversions = 0
        for i in range(self.total_n):
            for j in range(i + 1, self.total_n):
                x, y = self.status[i], self.status[j]
                num_inversions += int(x > y > 0)
        if self.n % 2 == 0:
            return (self.zero_pos // self.n + num_inversions) % 2 == 1, num_inversions
        else:
            return num_inversions % 2 == 0, num_inversions

    def solve(self, log=True) -> Optional[Tuple[int, int, List[np.ndarray]]]:
        close_list: Set[str] = set()
        open_list: List[PuzzleNode] = [self]
        open_list_map: Dict[str, PuzzleNode] = {PuzzleNode.get_fingerprint(self.status): self}
        n, num_nodes_check, num_nodes_expand = self.n, 0, 1

        def print_log(*args, **kwargs):
            if log:
                print(args, kwargs)

        def step(nx, ny):
            nonlocal num_nodes_expand
            new_pos = nx * n + ny
            # get the exchange status
            new_status = root.status.copy()
            tmp = new_status[zero_pos]
            new_status[zero_pos] = new_status[new_pos]
            new_status[new_pos] = tmp
            assert tmp == 0
            new_fingerprint = PuzzleNode.get_fingerprint(new_status)
            if new_fingerprint in close_list:
                return
            current_cost = root.current_cost + 1
            t_pos = root.status[new_pos] - 1
            tx, ty = t_pos // n, t_pos % n
            new_val = abs(zx - tx) + abs(zy - ty)
            old_val = abs(nx - tx) + abs(ny - ty)
            estimate_cost = root.estimate_cost + new_val - old_val
            if new_fingerprint in open_list_map:
                new_node = open_list_map[new_fingerprint]
                assert new_node.estimate_cost == estimate_cost
                if current_cost < new_node.current_cost:
                    new_node.parent, new_node.current_cost = root, current_cost
                    heapq.heapify(open_list)  # how to improve the efficiency here?
            else:
                new_node = PuzzleNode(n, status=new_status, parent=root,
                                      current_cost=current_cost, estimate_cost=estimate_cost, zero_pos=new_pos)
                heapq.heappush(open_list, new_node)
                open_list_map[new_fingerprint] = new_node
            num_nodes_expand += 1

        answer = None
        while len(open_list) > 0:
            root = heapq.heappop(open_list)
            root_fingerprint = PuzzleNode.get_fingerprint(root.status)
            del open_list_map[root_fingerprint]
            if root.estimate_cost == 0:
                # find the answer
                answer = root
                break
            # check for possible nodes
            zero_pos = root.zero_pos
            zx, zy = zero_pos // n, zero_pos % n
            if zx > 0:
                step(zx - 1, zy)
            if zx < n - 1:
                step(zx + 1, zy)
            if zy > 0:
                step(zx, zy - 1)
            if zy < n - 1:
                step(zx, zy + 1)
            close_list.add(root_fingerprint)
            num_nodes_check += 1

        print_log("Total nodes checked:", num_nodes_check)
        print_log("Total nodes expanded:", num_nodes_expand)
        if answer is None:
            print_log('No Solution!')
            return None
        else:
            solution_path = []
            cur_node = answer
            while cur_node is not None:
                solution_path.append(cur_node)
                cur_node = cur_node.parent
            solution_path.reverse()
            print_log("Total solution steps:", len(solution_path) - 1)
            print_log('The solution path is:')
            res_path = []
            for node in solution_path:
                if log:
                    node.print_status()
                res_path.append(np.array(node.status).reshape((self.n, self.n)))
                print_log('\n' + '=' * 30 + '\n')
            return num_nodes_check, num_nodes_expand, res_path
