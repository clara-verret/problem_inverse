import argparse

from global_constants import NUM_MC, R_MIN, R_MAX, MEAN, SIGMA
from analyze import analyze_direct_problem, analyze_discrete_inverse_problem, analyze_discrete_direct_problem

def main() :
    """Compute either the direct or inverse problem"""
    parser = argparse.ArgumentParser(description="parser used to chose which problem to solve")
    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group()

    #direct problem
    group.add_argument('-d', '--direct', action='store_true', help="solve direct problem")
    #inverse problem
    group.add_argument('-i', '--inverse', action='store_true', help="solve inverse problem")
    #adding noise to CLD
    parser.add_argument('-n', '--noise', action='store_true', help="add gaussian noise of mean 0 and sigma 0.01 to the CLD")


    args = parser.parse_args()
    if args.direct:
        #analyze_direct_problem(NUM_MC, R_MIN, R_MAX, MEAN, SIGMA)
        analyze_discrete_direct_problem(NUM_MC, R_MIN, R_MAX, MEAN, SIGMA)
    elif args.inverse:
        analyze_discrete_inverse_problem(NUM_MC, R_MIN, R_MAX, MEAN, SIGMA, args.noise)
       


if __name__ == '__main__':
    main()