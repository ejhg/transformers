public class LowRankSVD
{
    // Dot product of two vectors
    static double Dot (double[] a, double[] b) {
        double sum = 0;
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    // Matrix-vector multiplication
    static double[] MatVecMul (double[,] A, double[] x) {
        var rows = A.GetLength (0);
        var cols = A.GetLength (1);
        var result = new double[rows];
        for (var i = 0; i < rows; i++) {
            result[i] = 0;
            for (var j = 0; j < cols; j++)
                result[i] += A[i, j] * x[j];
        }

        return result;
    }

    // Transpose matrix-vector multiplication
    static double[] MatTransposeVecMul (double[,] A, double[] x) {
        var rows = A.GetLength (0);
        var cols = A.GetLength (1);
        var result = new double[cols];
        for (var j = 0; j < cols; j++) {
            result[j] = 0;
            for (var i = 0; i < rows; i++)
                result[j] += A[i, j] * x[i];
        }

        return result;
    }

    // Normalize a vector
    static void Normalize (double[] v) {
        var norm = Math.Sqrt (Dot (v, v));
        if (norm == 0) return;
        for (var i = 0; i < v.Length; i++) v[i] /= norm;
    }

    // Power iteration for largest singular value
    public static (double[] u, double[] v, double sigma) PowerIteration (double[,] A, int maxIter = 100, double tol = 1e-6) {
        var cols = A.GetLength (1);
        var rand = new Random ();

        var v = new double[cols];
        for (var i = 0; i < cols; i++) v[i] = rand.NextDouble ();

        Normalize (v);
        double sigma = 0;

        for (var iter = 0; iter < maxIter; iter++) {
            var Av = MatVecMul (A, v);
            var u = (double[])Av.Clone ();
            Normalize (u);

            var Atu = MatTransposeVecMul (A, u);
            v = (double[])Atu.Clone ();
            Normalize (v);

            var newSigma = Dot (u, Av);
            if (Math.Abs (newSigma - sigma) < tol) {
                break;
            }
            sigma = newSigma;
        }

        var finalU = MatVecMul (A, v);
        Normalize (finalU);
        return (finalU, v, sigma);
    }

    // Low-rank SVD approximation
    public static void LowRankDecompose (
        double[,] A,
        int k,
        out double[,] U,
        out double[] S,
        out double[,] Vt
    ) {
        var rows = A.GetLength (0);
        var cols = A.GetLength (1);

        U = new double[rows, k];
        Vt = new double[k, cols];
        S = new double[k];

        var residual = (double[,])A.Clone ();

        for (var i = 0; i < k; i++) {
            var (u, v, sigma) = PowerIteration (residual);

            for (var j = 0; j < rows; j++) U[j, i] = u[j];
            for (var j = 0; j < cols; j++) Vt[i, j] = v[j];
            S[i] = sigma;

            // Subtract rank-1 component: residual -= sigma * outer(u, v)
            for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                residual[r, c] -= sigma * u[r] * v[c];
        }
    }
}
