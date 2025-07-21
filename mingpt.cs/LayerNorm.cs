namespace mingpt.cs;

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
            double stdInv = 1.0 / Math.Sqrt (variance + epsilon);

            double[] dXHat = new double[D];
            double dVar = 0.0;
            double dMean = 0.0;

            for (int j = 0; j < D; j++) {
                double xHat = (Input.Data[i, j] - mean) * stdInv;
                GradGamma[j] += dOutput.Data[i, j] * xHat;
                GradBeta[j] += dOutput.Data[i, j];
                dXHat[j] = dOutput.Data[i, j] * Gamma[j];
            }

            for (int j = 0; j < D; j++) {
                dVar += dXHat[j] * (Input.Data[i, j] - mean) * -0.5 * Math.Pow (variance + epsilon, -1.5);
            }

            for (int j = 0; j < D; j++) {
                dMean += dXHat[j] * -stdInv + dVar * -2.0 * (Input.Data[i, j] - mean) / D;
            }

            for (int j = 0; j < D; j++) {
                dx.Data[i, j] =
                    dXHat[j] * stdInv +
                    dVar * 2.0 * (Input.Data[i, j] - mean) / D +
                    dMean / D;
            }
        }

        return dx;
    }

    public void UpdateParameters (double LearningRate) {
        for (int i = 0; i < EmbeddingSize; i++) {
            Gamma[i] -= LearningRate * GradGamma[i];
            Beta[i] -= LearningRate * GradBeta[i];
            GradGamma[i] = 0.0;
            GradBeta[i] = 0.0;
        }
    }
}
