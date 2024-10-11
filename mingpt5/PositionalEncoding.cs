namespace mingpt5;

public class PositionalEncoding
{
    public int EmbeddingDim;

    public PositionalEncoding (int embeddingDim) {
        EmbeddingDim = embeddingDim;
    }

    public Vector ApplyRoPE (Vector embedding, int position) {
        int halfDim = EmbeddingDim / 2;
        double theta = 10000.0;
        Vector output = embedding.Clone ();
        for (int i = 0; i < halfDim; i++) {
            double angle = position / Math.Pow (theta, (2.0 * i) / EmbeddingDim);
            double sinAngle = Math.Sin (angle);
            double cosAngle = Math.Cos (angle);
            double original1 = embedding.Data[2 * i];
            double original2 = embedding.Data[2 * i + 1];

            output.Data[2 * i] = original1 * cosAngle - original2 * sinAngle;
            output.Data[2 * i + 1] = original1 * sinAngle + original2 * cosAngle;
        }

        return output;
    }
}
