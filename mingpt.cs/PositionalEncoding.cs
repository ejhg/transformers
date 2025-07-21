namespace mingpt.cs;

public enum PositionalEncodingType
{
    Learnable,
    Sinusoidal
}

public class PositionalEncoding
{
    public PositionalEncodingType Type { get; private set; }
    public int MaxSeqLen { get; private set; }
    public int EmbeddingSize { get; private set; }
    
    // For learnable embeddings
    private EmbeddingLayer LearnableEmbeddings;
    
    // For sinusoidal embeddings
    private Matrix SinusoidalEmbeddings;

    public PositionalEncoding (int maxSeqLen, int embeddingSize, PositionalEncodingType type = PositionalEncodingType.Learnable) {
        Type = type;
        MaxSeqLen = maxSeqLen;
        EmbeddingSize = embeddingSize;

        if (type == PositionalEncodingType.Learnable) {
            LearnableEmbeddings = new EmbeddingLayer (maxSeqLen, embeddingSize);
        } else {
            CreateSinusoidalEmbeddings ();
        }
    }

    private void CreateSinusoidalEmbeddings () {
        SinusoidalEmbeddings = new Matrix (MaxSeqLen, EmbeddingSize);
        
        for (int pos = 0; pos < MaxSeqLen; pos++) {
            for (int i = 0; i < EmbeddingSize; i++) {
                double angle = GetAngle (pos, i);
                if (i % 2 == 0) {
                    SinusoidalEmbeddings.Data[pos, i] = Math.Sin (angle);
                } else {
                    SinusoidalEmbeddings.Data[pos, i] = Math.Cos (angle);
                }
            }
        }
    }

    private double GetAngle (int pos, int i) {
        double exponent = (double)(2 * (i / 2)) / EmbeddingSize;
        return pos / Math.Pow (10000, exponent);
    }

    public Matrix Forward (int[] positions) {
        if (Type == PositionalEncodingType.Learnable) {
            return LearnableEmbeddings.Forward (positions);
        } else {
            var result = new Matrix (positions.Length, EmbeddingSize);
            for (int i = 0; i < positions.Length; i++) {
                int pos = positions[i];
                for (int j = 0; j < EmbeddingSize; j++) {
                    result.Data[i, j] = SinusoidalEmbeddings.Data[pos, j];
                }
            }
            return result;
        }
    }

    public void Backward (Matrix dPositionalEmbeddings, int[] positions) {
        if (Type == PositionalEncodingType.Learnable) {
            LearnableEmbeddings.Backward (dPositionalEmbeddings, positions);
        }
        // Sinusoidal embeddings don't have learnable parameters, so no backward pass needed
    }

    public void UpdateParameters (double learningRate) {
        if (Type == PositionalEncodingType.Learnable) {
            LearnableEmbeddings.UpdateParameters (learningRate);
        }
        // Sinusoidal embeddings don't have learnable parameters
    }
}