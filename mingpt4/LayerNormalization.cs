using mingpt5;

namespace mingpt4;

class LayerNormalization
{
    public int EmbeddingDim { get; set; }
    public Vector Gamma { get; set; }
    public Vector Beta { get; set; }
    public Vector dGamma { get; set; }
    public Vector dBeta { get; set; }

    public LayerNormalization (int embeddingDim) {
        EmbeddingDim = embeddingDim;
        Gamma = new Vector (embeddingDim);
        Beta = new Vector (embeddingDim);
        dGamma = new Vector (embeddingDim);
        dBeta = new Vector (embeddingDim);

        for (int i = 0; i < EmbeddingDim; i++)
            Gamma.Data[i] = 1.0; // Initialize gamma to 1
    }

    public Matrix Forward (Matrix X) {
        int seqLength = X.Rows;
        Matrix normalized = new Matrix (seqLength, EmbeddingDim);

        for (int i = 0; i < seqLength; i++) {
            double mean = 0;
            for (int j = 0; j < EmbeddingDim; j++)
                mean += X.Data[i][j];
            mean /= EmbeddingDim;

            double variance = 0;
            for (int j = 0; j < EmbeddingDim; j++)
                variance += Math.Pow (X.Data[i][j] - mean, 2);
            variance /= EmbeddingDim;

            double std = Math.Sqrt (variance + 1e-6);

            for (int j = 0; j < EmbeddingDim; j++) {
                double normalizedValue = (X.Data[i][j] - mean) / std;
                normalized.Data[i][j] = Gamma.Data[j] * normalizedValue + Beta.Data[j];
            }
        }

        return normalized;
    }

    // Backward pass (omitted for brevity)
    // You would compute gradients w.r.t Gamma and Beta here

    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < EmbeddingDim; i++) {
            Gamma.Data[i] -= learningRate * dGamma.Data[i];
            Beta.Data[i] -= learningRate * dBeta.Data[i];
            dGamma.Data[i] = 0;
            dBeta.Data[i] = 0;
        }
    }
}
