using NUnit.Framework;

namespace transformers.tests.svd;

public class LowRankSVDTests
{
    static void PrintMatrix (double[,] M) {
        int rows = M.GetLength (0);
        int cols = M.GetLength (1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                Console.Write ($"{M[i, j]:0.000} ");
            Console.WriteLine ();
        }
        Console.WriteLine ();
    }

    [Test]
    public void test2 () {
        const int rows = 5;
        const int cols = 4;
        const int rank = 2;
        const double epsilon = 1e-4;

        // Generate a random matrix A
        var rand = new Random (42);
        double[,] A = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            A[i, j] = rand.NextDouble () * 2 - 1; // values in [-1, 1]

        PrintMatrix (A);

        // Decompose A
        LowRankSVD.LowRankDecompose (A, rank, out var U, out var S, out var Vt);

        // Reconstruct A_approx = U * diag(S) * Vt
        double[,] S_diag = new double[rank, rank];
        for (int i = 0; i < rank; i++)
            S_diag[i, i] = S[i];

        double[,] US = Multiply (U, S_diag); // U * S
        double[,] A_approx = Multiply (US, Vt); // (U * S) * Vᵀ

        PrintMatrix (A_approx);

        // Compute reconstruction error
        double maxDiff = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = Math.Abs (A[i, j] - A_approx[i, j]);
                if (diff > maxDiff) maxDiff = diff;
            }
        }

        Assert.LessOrEqual (maxDiff, epsilon);
    }

    // Matrix multiplication helper
    private static double[,] Multiply (double[,] A, double[,] B) {
        int m = A.GetLength (0);
        int n = A.GetLength (1);
        int p = B.GetLength (1);
        double[,] result = new double[m, p];
        for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
        for (int k = 0; k < n; k++)
            result[i, j] += A[i, k] * B[k, j];
        return result;
    }
}
