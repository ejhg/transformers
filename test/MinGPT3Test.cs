using transformers.utils;

namespace mingpt3;

public class MinGPT3Test
{
    public static void run () {
        int embeddingSize = 96;
        int numHeads = 6;
        int numLayers = 4;
        int maxSeqLen = 8;
        int batchSize = 8;

        var data = DataLoader.LoadData (maxSeqLen, out var vocabSize, out var vocabulary);

        var model = new Model (vocabSize, embeddingSize, numHeads, numLayers, maxSeqLen);
        var optimizer = new Optimizer (learningRate: 0.0005);

        Trainer.train (
            model,
            optimizer,
            batchSize,
            data,
            vocabulary);
    }
}
