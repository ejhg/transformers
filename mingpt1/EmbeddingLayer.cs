using mingpt3;

namespace mingpt1;

public class EmbeddingLayer
{
    public int VocabSize, EmbeddingSize;
    public Matrix Weights;
    public Matrix GradWeights;
    private int[] InputIds;

    public EmbeddingLayer (int vocabSize, int embeddingSize) {
        VocabSize = vocabSize;
        EmbeddingSize = embeddingSize;
        Weights = Matrix.Random (vocabSize, embeddingSize);
        GradWeights = new Matrix (vocabSize, embeddingSize);
    }

    public Matrix Forward (int[] inputIds) {
        InputIds = inputIds;
        var result = new Matrix (inputIds.Length, EmbeddingSize);
        for (int i = 0; i < inputIds.Length; i++)
        for (int j = 0; j < EmbeddingSize; j++)
            result.Data[i, j] = Weights.Data[inputIds[i], j];
        return result;
    }

    public void Backward (Matrix dOutput) {
        for (int i = 0; i < InputIds.Length; i++)
        for (int j = 0; j < EmbeddingSize; j++)
            GradWeights.Data[InputIds[i], j] += dOutput.Data[i, j];
    }
}
