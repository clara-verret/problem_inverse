import argparse

from direct_problem import solve_direct_problem_normal_psd
from global_constants import NUM_MC, R_MIN, R_MAX, MEAN, SIGMA
from analyze import analyze_direct_problem

def main() :
    """Compute either the direct or inverse problem"""
    parser = argparse.ArgumentParser(description="parser used to chose which problem to solve")
    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group()

    #direct problem
    group.add_argument('-d', '--direct', action='store_true', help="solve direct problem")
    #inverse problem
    group.add_argument('-i', '--inverse', action='store_true', help="solve inverse problem")

    args = parser.parse_args()
    if args.direct:
        print("Analysing direct problem")
        analyze_direct_problem(NUM_MC, R_MIN, R_MAX, MEAN, SIGMA)
    elif args.inverse:
        print("Solving inverse problem")
        #mettre la fonction qui inverse le problème


if __name__ == '__main__':
    main()