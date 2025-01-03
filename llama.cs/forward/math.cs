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

    public static unsafe void MatMul (float[,] xout, float[] x, float[,] W) {
        var size = xout.GetLength (1);

        if (x.Length > 1000) {
            Parallel.For (0, xout.Length, i => {
                int n = x.Length;

                fixed (float* pW = W) {
                    float val = 0.0f;
                    float* pRowW = pW + i * n;

                    for (int j = 0; j < n; j++) {
                        val += pRowW[j] * x[j];
                    }

                    xout[i / size, i % size] = val;
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

                    xout[i / size, i % size] = val;
                }
            }
        }
    }
}
