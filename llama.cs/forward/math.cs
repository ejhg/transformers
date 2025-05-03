namespace llama.cs;

public class math
{
    public static void RmsNorm (float[] o, float[] x, float[] weight) {
        var size = x.Length;

        // Calculate sum of squares
        float sumOfSquaresOfX = 0.0f;
        for (int j = 0; j < size; j++) {
            sumOfSquaresOfX += x[j] * x[j];
        }

        var scaleX = 1.0f / MathF.Sqrt (sumOfSquaresOfX / size + 1e-5f);

        // Normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (scaleX * x[j]);
        }
    }

    public static void Softmax (float[] x, int size) {
        // Find max value
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }

        // Exponentiate and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = MathF.Exp (x[i] - max_val);
            sum += x[i];
        }

        // Normalize
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }

    /**
     * W (m,n) @ x (n,) -> xout (m,)
     */
    public static void MatMul (float[] xout, float[] x, float[][] W) {
        if (x.Length > 1000) {
            Parallel.For (0, xout.Length, i => {
                float val = 0.0f;
                int n = x.Length;
                var row = W[i];

                for (int j = 0; j < n; j++) {
                    val += row[j] * x[j];
                }

                xout[i] = val;
            });
        } else {
            for (var i = 0; i < xout.Length; i++) {
                float val = 0.0f;
                int n = x.Length;
                var row = W[i];

                for (int j = 0; j < n; j++) {
                    val += row[j] * x[j];
                }

                xout[i] = val;
            }
        }
    }

    /**
     * W (m,n) @ x (n,) -> xout (m,)
     * Standard matrix multiplication
     */
    public static unsafe void MatMul (float[] xout, float[] x, float[,] W) {
        if (x.Length > 1000) {
            Parallel.For (0, xout.Length, i => {
                int n = x.Length;

                fixed (float* pW = W) {
                    float val = 0.0f;
                    float* pRowW = pW + i * n;

                    for (int j = 0; j < n; j++) {
                        val += pRowW[j] * x[j];
                    }

                    xout[i] = val;
                }
            });
        } else {
            for (var i = 0; i < xout.Length; i++) {
                int n = x.Length;

                fixed (float* pW = W) {
                    float val = 0.0f;
                    float* pRowW = pW + i * n;

                    for (int j = 0; j < n; j++) {
                        val += pRowW[j] * x[j];
                    }

                    xout[i] = val;
                }
            }
        }
    }
    
    /**
     * SVD Matrix multiplication: (U * diag(S) * Vt) @ x = U * (diag(S) * (Vt @ x))
     * This computes the matrix multiplication using the decomposed form,
     * which is more efficient for large matrices with low rank decomposition.
     */
    public static void MatMulSVD(float[] xout, float[] x, weights.SVDMatrix svdMatrix) {
        if (!svdMatrix.use_svd || svdMatrix.U == null || svdMatrix.S == null || svdMatrix.Vt == null) {
            // Fall back to standard matrix multiplication if not using SVD or any component is null
            if (svdMatrix.original != null) {
                MatMul(xout, x, svdMatrix.original);
            } else {
                // This should never happen if properly initialized
                throw new InvalidOperationException("SVD matrix is not properly initialized");
            }
            return;
        }
        
        var rank = svdMatrix.rank;
        var rows = svdMatrix.U.GetLength(0);
        var cols = svdMatrix.Vt.GetLength(1);
        
        // Step 1: Compute Vt @ x (resulting in a vector of size rank)
        var temp = new float[rank];
        
        if (cols > 1000) {
            Parallel.For(0, rank, i => {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++) {
                    sum += svdMatrix.Vt[i, j] * x[j];
                }
                temp[i] = sum;
            });
        } else {
            for (int i = 0; i < rank; i++) {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++) {
                    sum += svdMatrix.Vt[i, j] * x[j];
                }
                temp[i] = sum;
            }
        }
        
        // Step 2: Scale by singular values
        for (int i = 0; i < rank; i++) {
            temp[i] *= svdMatrix.S[i];
        }
        
        // Step 3: Compute U @ (diag(S) * (Vt @ x))
        if (rows > 1000) {
            Parallel.For(0, rows, i => {
                float sum = 0.0f;
                for (int j = 0; j < rank; j++) {
                    sum += svdMatrix.U[i, j] * temp[j];
                }
                xout[i] = sum;
            });
        } else {
            for (int i = 0; i < rows; i++) {
                float sum = 0.0f;
                for (int j = 0; j < rank; j++) {
                    sum += svdMatrix.U[i, j] * temp[j];
                }
                xout[i] = sum;
            }
        }
    }
    
    // Removed the MatMul with SVDMatrix overload - we'll call MatMulSVD directly

    public static unsafe void MatMul (float[,] xout, float[] x, float[,] W) {
        var out_size = xout.GetLength (1);
        var h = W.GetLength (0);
        var w = W.GetLength (1);

        if (x.Length > 1000) {
            Parallel.For (0, h, i => {
                fixed (float* pW = W) {
                    float val = 0.0f;
                    float* pRowW = pW + i * w;

                    for (int j = 0; j < w; j++) {
                        val += pRowW[j] * x[j];
                    }

                    xout[i / out_size, i % out_size] = val;
                }
            });
        } else {
            for (var i = 0; i < h; i++) {
                float val = 0.0f;

                for (int j = 0; j < w; j++) {
                    val += W[i, j] * x[j];
                }

                xout[i / out_size, i % out_size] = val;
            }
        }
    }
}
