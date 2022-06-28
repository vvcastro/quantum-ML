from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dimod import ExactSolver, SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel
import numpy as np

class QuantumRegressor:

    samplers = {
        'hybrid': LeapHybridSampler,
        'simulated': SimulatedAnnealingSampler,
        'cpu': ExactSolver,
        '2000Q': lambda: EmbeddingComposite(DWaveSampler(solver=dict(topology__type='chimera'))),
        'advantage': lambda: EmbeddingComposite(DWaveSampler(solver=dict(topology__type='pegasus')))
    }
    
    def __init__(self, sampler, precision=3):
        
        # define problem formulation and sampler
        self.bqm = BinaryQuadraticModel('BINARY', offset=0)
        self.sampler = self.samplers[sampler]()

        # define a precision vector for the approximations
        precision_values = [2 ** -(p + 1) for p in range(precision)]
        self.precision_vector = np.array(precision_values[::-1]).T
    
    def build_problem(self, X, Y, precision=3):
        self.X, self.Y = X, Y

        # augmentate X feature dim to predict the intercept
        feature_dim = X.shape[1]
        
        # define precision matrix
        self.P_matrix = np.kron(np.identity(feature_dim), self.precision_vector.T)
        
        # compute initial A matrix and b vector
        A_matrix = self.P_matrix.T.dot(X.T).dot(X).dot(self.P_matrix)
        b_vector = -2 * self.P_matrix.T.dot(X.T).dot(Y)
        
        # operate to convert the problem to the correct form
        upper_triangular_A_matrix = self.to_upper_triangular_matrix(A_matrix)
        A_matrix_diag = np.diagonal(upper_triangular_A_matrix).reshape(-1, 1).copy()
        np.fill_diagonal(upper_triangular_A_matrix, 0)
        
        # define the final matrix and vector values
        self.A_matrix = upper_triangular_A_matrix
        self.b_vector = b_vector + A_matrix_diag

        # initialize solver with values
        self.bqm.add_quadratic_from_dense(self.A_matrix)
        self.bqm.add_linear_from_array(self.b_vector.ravel())
        return "Ready to fit"

    def solve(self, label=None):
        
        # get the sample results
        self.sampleset = self.sampler.sample(bqm=self.bqm, label=label)
        samples, energies = self.sampleset.record.sample, self.sampleset.record.energy

        # bias: the best solution is the solution with lowest energy and less 1 variables
        best_solutions = samples[np.where(energies == energies.min())]
        self.solution = best_solutions[ best_solutions.sum(axis=1).argmin() ]

        print(self.solution)
        self.predicted_slope = np.dot(self.solution, self.P_matrix.T)[0]
        self.predicted_intercept = (self.Y.mean() - (self.X.mean(axis=0)).dot(self.predicted_slope))[0]
        print("Predicted Slope:", self.predicted_slope)
        print("Predicted Intercept:", self.predicted_intercept)
        return "Done!"

    @staticmethod
    def to_upper_triangular_matrix(matrix):
        output = np.zeros_like(matrix)
        for i in range(matrix.shape[0]): # rows
            for j in range(matrix.shape[1]): # cols
                if (i < j):
                    output[i, j] = matrix[i, j] + matrix[j, i]
                elif (i == j):
                    output[i, j] = matrix[i, j]
        return output
