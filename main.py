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

    puz = PuzzleNode(args.n, log=True)
    puz.check_solvable()
    print()
    puz.solve()

    fout.close()


if __name__ == '__main__':
    main()
