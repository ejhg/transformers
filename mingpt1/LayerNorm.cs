using mingpt3;

namespace mingpt1;

public class LayerNorm
{
    public int EmbeddingSize;
    public double[] Gamma;
    public double[] Beta;
    public double[] GradGamma;
    public double[] GradBeta;
    private double[] Mean;
    private double[] Variance;
    private Matrix Input;

    public LayerNorm (int embeddingSize) {
        EmbeddingSize = embeddingSize;
        Gamma = new double[embeddingSize];
        Beta = new double[embeddingSize];
        GradGamma = new double[embeddingSize];
        GradBeta = new double[embeddingSize];
        for (int i = 0; i < embeddingSize; i++)
            Gamma[i] = 1.0;
    }

    public Matrix Forward (Matrix x) {
        Input = x;
        int N = x.Rows;
        int D = x.Cols;
        Mean = new double[N];
        Variance = new double[N];
        var result = new Matrix (N, D);
        double epsilon = 1e-5;

        for (int i = 0; i < N; i++) {
            double mean = 0.0;
            for (int j = 0; j < D; j++)
                mean += x.Data[i, j];
            mean /= D;
            Mean[i] = mean;

            double variance = 0.0;
            for (int j = 0; j < D; j++)
                variance += (x.Data[i, j] - mean) * (x.Data[i, j] - mean);
            variance /= D;
            Variance[i] = variance;

            for (int j = 0; j < D; j++) {
                double normalized = (x.Data[i, j] - mean) / Math.Sqrt (variance + epsilon);
                result.Data[i, j] = Gamma[j] * normalized + Beta[j];
            }
        }

        return result;
    }

    public Matrix Backward (Matrix dOutput) {
        int N = dOutput.Rows;
        int D = dOutput.Cols;
        var dx = new Matrix (N, D);
        double epsilon = 1e-5;

        for (int i = 0; i < N; i++) {
            double mean = Mean[i];
            double variance = Variance[i];

            for (int j = 0; j < D; j++) {
                double x_hat = (Input.Data[i, j] - mean) / Math.Sqrt (variance + epsilon);
                GradGamma[j] += dOutput.Data[i, j] * x_hat;
                GradBeta[j] += dOutput.Data[i, j];
            }

            for (int j = 0; j < D; j++) {
                double x_hat = (Input.Data[i, j] - mean) / Math.Sqrt (variance + epsilon);
                double d_xhat = dOutput.Data[i, j] * Gamma[j];

                // Compute gradients
                double dvar = -0.5 * d_xhat * x_hat / (variance + epsilon);
                double dmean = -d_xhat / Math.Sqrt (variance + epsilon);
                for (int k = 0; k < D; k++) {
                    dx.Data[i, k] += d_xhat / Math.Sqrt (variance + epsilon);
                    dx.Data[i, k] += 2.0 * (Input.Data[i, k] - mean) * dvar / D;
                    dx.Data[i, k] += dmean / D;
                }
            }
        }

        return dx;
    }
}
