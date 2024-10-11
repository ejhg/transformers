using mingpt5;

namespace mingpt4;

class EmbeddingLayer
{
    private mingpt3.EmbeddingLayer embedding;

    public EmbeddingLayer (int vocabSize, int embeddingSize) {
        embedding = new mingpt3.EmbeddingLayer (vocabSize, embeddingSize);
    }

    public Matrix Forward (int[] inputTokens) {
        var ret = embedding.Forward (inputTokens);
        return new Matrix (ret.Data);
    }

    public void UpdateParameters (double learningRate) {
        embedding.UpdateParameters (learningRate);
    }
}
