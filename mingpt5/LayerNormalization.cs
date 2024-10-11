namespace mingpt5;

public class LayerNormalization
{
    public int EmbeddingDim;
    public Vector Gamma;
    public Vector Beta;
    public Vector Input;
    public Vector Normalized;
    public double Mean;
    public double Variance;

    public LayerNormalization (int embeddingDim) {
        EmbeddingDim = embeddingDim;
        Gamma = new Vector (embeddingDim);
        Beta = new Vector (embeddingDim);

        for (int i = 0; i < EmbeddingDim; i++) {
            Gamma.Data[i] = 1.0;
        }
    }

    public Vector Forward (Vector input) {
        Input = input.Clone ();
        Mean = 0.0;
        Variance = 0.0;

        for (int i = 0; i < EmbeddingDim; i++) {
            Mean += Input.Data[i];
        }

        Mean /= EmbeddingDim;

        for (int i = 0; i < EmbeddingDim; i++) {
            Variance += Math.Pow (Input.Data[i] - Mean, 2);
        }

        Variance /= EmbeddingDim;

        Normalized = new Vector (EmbeddingDim);
        for (int i = 0; i < EmbeddingDim; i++) {
            Normalized.Data[i] = (Input.Data[i] - Mean) / Math.Sqrt (Variance + 1e-6);
            Normalized.Data[i] = Gamma.Data[i] * Normalized.Data[i] + Beta.Data[i];
        }

        return Normalized;
    }

    public Vector Backward (Vector dout) {
        Vector dGamma = new Vector (EmbeddingDim);
        Vector dBeta = new Vector (EmbeddingDim);
        Vector dx = new Vector (EmbeddingDim);

        double invStd = 1.0 / Math.Sqrt (Variance + 1e-6);
        Vector xHat = new Vector (EmbeddingDim);
        for (int i = 0; i < EmbeddingDim; i++) {
            xHat.Data[i] = (Input.Data[i] - Mean) * invStd;
        }

        for (int i = 0; i < EmbeddingDim; i++) {
            dGamma.Data[i] += dout.Data[i] * xHat.Data[i];
            dBeta.Data[i] += dout.Data[i];
        }

        for (int i = 0; i < EmbeddingDim; i++) {
            double dXhat = dout.Data[i] * Gamma.Data[i];
            double dVar = -0.5 * dXhat * (Input.Data[i] - Mean) * Math.Pow (Variance + 1e-6, -1.5);
            double dMean = -dXhat * invStd;
            dx.Data[i] += dXhat * invStd + dVar * 2.0 * (Input.Data[i] - Mean) / EmbeddingDim + dMean / EmbeddingDim;
        }

        // Update parameters
        for (int i = 0; i < EmbeddingDim; i++) {
            Gamma.Data[i] -= dout.Data[i] * xHat.Data[i];
            Beta.Data[i] -= dout.Data[i];
        }

        return dx;
    }
}
