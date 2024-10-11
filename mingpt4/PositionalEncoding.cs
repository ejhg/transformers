using mingpt5;

namespace mingpt4;

class PositionalEncoding
{
    public int MaxSeqLength { get; set; }
    public int EmbeddingDim { get; set; }
    public Matrix PositionalEncodings { get; set; }

    public PositionalEncoding (int maxSeqLength, int embeddingDim) {
        MaxSeqLength = maxSeqLength;
        EmbeddingDim = embeddingDim;
        PositionalEncodings = new Matrix (MaxSeqLength, EmbeddingDim);

        for (int pos = 0; pos < MaxSeqLength; pos++) {
            for (int i = 0; i < EmbeddingDim; i++) {
                double angle = GetAngle (pos, i);
                if (i % 2 == 0)
                    PositionalEncodings.Data[pos][i] = Math.Sin (angle);
                else
                    PositionalEncodings.Data[pos][i] = Math.Cos (angle);
            }
        }
    }

    private double GetAngle (int pos, int i) {
        double exponent = (double)(2 * (i / 2)) / EmbeddingDim;
        return pos / Math.Pow (10000, exponent);
    }

    public Matrix AddPositionalEncoding (Matrix embeddings) {
        int seqLength = embeddings.Rows;
        Matrix result = new Matrix (seqLength, EmbeddingDim);
        for (int i = 0; i < seqLength; i++)
        for (int j = 0; j < EmbeddingDim; j++)
            result.Data[i][j] = embeddings.Data[i][j] + PositionalEncodings.Data[i][j];
        return result;
    }
}
