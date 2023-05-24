import sys
import random
import argparse
from puzzle import PuzzleNode


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
    main()
