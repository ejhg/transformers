namespace llama.cs;

public class LowRankSVD
{
    // Dot product of two vectors
    public static float Dot(float[] a, float[] b) {
        float sum = 0;
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    // Matrix-vector multiplication
    public static float[] MatVecMul(float[,] A, float[] x) {
        var rows = A.GetLength(0);
        var cols = A.GetLength(1);
        var result = new float[rows];
        for (var i = 0; i < rows; i++) {
            result[i] = 0;
            for (var j = 0; j < cols; j++)
                result[i] += A[i, j] * x[j];
        }

        return result;
    }

    // Transpose matrix-vector multiplication
    public static float[] MatTransposeVecMul(float[,] A, float[] x) {
        var rows = A.GetLength(0);
        var cols = A.GetLength(1);
        var result = new float[cols];
        for (var j = 0; j < cols; j++) {
            result[j] = 0;
            for (var i = 0; i < rows; i++)
                result[j] += A[i, j] * x[i];
        }

        return result;
    }

    // Normalize a vector
    public static void Normalize(float[] v) {
        var norm = MathF.Sqrt(Dot(v, v));
        if (norm == 0) return;
        for (var i = 0; i < v.Length; i++) v[i] /= norm;
    }

    // Power iteration for largest singular value
    public static (float[] u, float[] v, float sigma) PowerIteration(float[,] A, int maxIter = 100, float tol = 1e-6f) {
        var cols = A.GetLength(1);
        var rand = new Random();

        var v = new float[cols];
        for (var i = 0; i < cols; i++) v[i] = (float)rand.NextDouble();

        Normalize(v);
        float sigma = 0;

        for (var iter = 0; iter < maxIter; iter++) {
            var Av = MatVecMul(A, v);
            var u = (float[])Av.Clone();
            Normalize(u);

            var Atu = MatTransposeVecMul(A, u);
            v = (float[])Atu.Clone();
            Normalize(v);

            var newSigma = Dot(u, Av);
            if (MathF.Abs(newSigma - sigma) < tol) {
                break;
            }
            sigma = newSigma;
        }

        var finalU = MatVecMul(A, v);
        Normalize(finalU);
        return (finalU, v, sigma);
    }

    // Low-rank SVD approximation
    public static void LowRankDecompose(
        float[,] A,
        int k,
        out float[,] U,
        out float[] S,
        out float[,] Vt
    ) {
        var rows = A.GetLength(0);
        var cols = A.GetLength(1);

        U = new float[rows, k];
        Vt = new float[k, cols];
        S = new float[k];

        var residual = (float[,])A.Clone();

        for (var i = 0; i < k; i++) {
            var (u, v, sigma) = PowerIteration(residual);

            for (var j = 0; j < rows; j++) U[j, i] = u[j];
            for (var j = 0; j < cols; j++) Vt[i, j] = v[j];
            S[i] = sigma;

            // Subtract rank-1 component: residual -= sigma * outer(u, v)
            for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                residual[r, c] -= sigma * u[r] * v[c];
        }
    }

    // Calculate the Frobenius norm (measure of difference between original and decomposed matrix)
    public static float FrobeniusNorm(float[,] A, float[,] U, float[] S, float[,] Vt) {
        var rows = A.GetLength(0);
        var cols = A.GetLength(1);
        var k = S.Length;
        
        float sumSquaredDiff = 0;
        
        // Calculate reconstructed = U * diag(S) * Vt
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float reconstructed = 0;
                for (int l = 0; l < k; l++) {
                    reconstructed += U[i, l] * S[l] * Vt[l, j];
                }
                float diff = A[i, j] - reconstructed;
                sumSquaredDiff += diff * diff;
            }
        }
        
        return MathF.Sqrt(sumSquaredDiff);
    }
    
    // Helper for determining appropriate rank
    public static int DetermineRank(float[,] A, float errorThreshold = 0.01f, int maxRank = 50) {
        int rows = A.GetLength(0);
        int cols = A.GetLength(1);
        int maxPossibleRank = Math.Min(rows, cols);
        int rank = Math.Min(maxPossibleRank, maxRank);
        
        // Get singular values
        var (_, _, sigma) = PowerIteration(A);
        float totalEnergy = sigma * sigma;  // First singular value squared
        
        // If the matrix is small or close to identity, no need for SVD decomposition
        if (totalEnergy < 10.0f || rows * cols < 10000) {
            return 0;  // Don't use SVD for this matrix
        }
        
        // Start with a reasonable initial rank based on matrix size
        int initialRank = Math.Min(Math.Max(5, (int)(Math.Min(rows, cols) * 0.1f)), rank);
        return initialRank;
    }
}