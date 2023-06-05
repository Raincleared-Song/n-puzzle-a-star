import sys
import random
import argparse
import numpy as np
from typing import List, Optional, Tuple
from puzzle import PuzzleNode


def frontend_solve(inputs: np.ndarray) -> Optional[Tuple[int, int, List[np.ndarray]]]:
    """
    :param inputs: the initial state, shape [n, n], numbered 0~(n^2-1)
    :return: if solvable, return (search step count, expanded node count, the solution path)
             The solution path includes the initial state and the target state
             if unsolvable, return None
    """
    shape = inputs.shape
    assert len(shape) == 2 and shape[0] == shape[1]
    n = shape[0]
    puz = PuzzleNode(n, status=inputs.reshape(-1).tolist(), log=False)
    solvable, num_inversions = puz.check_solvable()
    if solvable:
        return puz.solve(log=False)
    else:
        return None


def batch_test():
    """
    test result:
    [0.2, 0.4, 0.6, 0.8, 1.03, 1.5, 1.75, 2.24, 2.96, 3.27,
     4.72, 7.75, 6.72, 11.65, 17.36, 16.29, 24.79, 27.0, 50.33, 40.01]
    average search step: 11.0685
    """
    test_case_dir = 'boards'
    num_stages, cases_per_stage = 20, 100
    counter = [0 for _ in range(num_stages)]
    for stage in range(num_stages):
        for case in range(num_stages):
            case_dir = f'{test_case_dir}/{stage}_{case}.npy'
            inputs = np.load(case_dir)
            res = frontend_solve(inputs)
            assert res is not None
            num_search_step, num_expanded_node, solution_path = res
            counter[stage] += num_search_step
        counter[stage] = round(counter[stage] / cases_per_stage, 2)
    print(counter)
    print('average search step:', sum(counter) / num_stages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', help='the value of n', type=int, default=3)
    parser.add_argument('--seed', help='the random seed', type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    fout = open(f'puzzle_{args.n}.txt', 'w')
    sys.stdout = fout

    # random initialization
    puz = PuzzleNode(args.n, log=True)
    # initialization with specific status, 0 represents the blank position
    # specific_status = [4, 1, 8, 0, 2, 7, 5, 3, 6]
    # puz = PuzzleNode(3, status=specific_status, log=True)

    solvable, num_inversions = puz.check_solvable()
    suffix = "SOLVABLE" if solvable else "UNSOLVABLE"
    print(f"Number of inversions is {num_inversions}, N is {args.n} and the blank is in row {puz.zero_pos // puz.n},"
          f" therefore it is {suffix} !!!\n")
    if solvable:
        puz.solve()
    else:
        print('No Solution!')

    fout.close()


if __name__ == '__main__':
    # main()
    batch_test()
