namespace mingpt.cs;

public class EmbeddingLayer
{
    public int VocabSize, EmbeddingSize;
    public Matrix Weights;
    public Matrix GradWeights;

    public EmbeddingLayer (int vocabSize, int embeddingSize) {
        VocabSize = vocabSize;
        EmbeddingSize = embeddingSize;
        Weights = Matrix.Random (vocabSize, embeddingSize);
        GradWeights = new Matrix (vocabSize, embeddingSize);
    }

    public Matrix Forward (int[] inputIds) {
        var result = new Matrix (inputIds.Length, EmbeddingSize);
        for (int i = 0; i < inputIds.Length; i++)
        for (int j = 0; j < EmbeddingSize; j++)
            result.Data[i, j] = Weights.Data[inputIds[i], j];
        return result;
    }

    public void Backward (Matrix dOutput, int[] inputIds) {
        for (int i = 0; i < inputIds.Length; i++)
        for (int j = 0; j < EmbeddingSize; j++)
            GradWeights.Data[inputIds[i], j] += dOutput.Data[i, j];
    }

    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < VocabSize; i++) {
            for (int j = 0; j < EmbeddingSize; j++) {
                Weights.Data[i, j] -= learningRate * GradWeights.Data[i, j];
                GradWeights.Data[i, j] = 0; // Reset gradients
            }
        }
    }
}
