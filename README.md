# Inferring particule size distribution with lasers
This project compute the direct and the inverse problem from https://arxiv.org/abs/2012.08287 .

## Getting started
To start using the code :

1. Clone the repository:
    ```bash
    git@github.com:clara-verret/problem_inverse.git
    ```
2. Run the application. 
   To run the direct problem (i.e to compute the CLD and k) use :
    ```bash
    python3 main.py -d
    ```
    Warning : it will run the direct problem for dirac PSD (uncomment the other lines to see the direct problem for normal functions).
    To run the inverse problem (i.e to get the PSD) use :
    ```bash
    python3 main.py -i
    ```
    To add noise in the inverse problem use :
    ```bash
    python3 main.py -i -n
    ```

## Some explanation

All global variables are defined in global.constants.py.

The file analyze.py save plots that can be seen in the Graphs or MeshPlots folder.

The file direct_problem.py compute the direct problem for all these options:
1. in the case of normal or dirac PSD
2. in the case of spherical or ellipsoid or polyhedrons particles
3. in the case of discretized or continuous kernel functions

The file inverse_problem.py compute the inverse problem in the case of discretized or continuous CLD.

The file mesh.py studies the case of polyhedrons

The file procrustes_distance.py present an elementary example of calculation of the Full Procrustes Distance. Note that the result depends on the order of the landmarks (here, two consecutive points correspond to an edge).

### Monte Carlo in the direct problem (see direct_problem.py)

We use Monte Carlo to estimate k, which allows us to compute the CLD in the direct problem.

#### For spherical particles

To model a random chord measurement by a laser, we randomly choose d in [0,1] (unit cirle) and compute the chord following this formula : $L = 2*\sqrt{1-d^2}$.
Then, for a circle of radius r the corresponding chord is $L' =r*L$.

#### For polyhedrons particles

For polyhedrons, we start by building a polyhedron with random orientation, then we project it in the x-y plane, then we choose a y such that the line y=y goes through the projection. The chord length is the length of the intersection between the y=y line and the projection (polygon).

### Hyper-parameter tuning for the inverse problem (see inverse_problem.py)
We recall the inverse problem : $\text{Find } \psi \in X \text{ minimizing } ||\mathcal{K} \psi - \overline{Q}||^2 + \delta ||\psi||^2$, where $\delta$ is an hyper-parameter to be tuned.
We use cross-validation to find its value : we create a train set by computing multiple normal functions of different means and standard deviations (which represent multiple PSD).
Then, we compute the corresponding CLD.
We inverse the problem for each of these (CLD,PSD) and find the $\delta$ that fit the best the data.
