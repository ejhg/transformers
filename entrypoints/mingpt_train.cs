using transformers.utils;

namespace mingpt3;

static class mingpt_train
{
    public static void run (string command) {
        train ("resources/tinyshakespeare.txt");
    }

    public static void train (string sourceFile) {
        int embeddingSize = 128;
        int numHeads = 6;
        int numLayers = 4;
        int maxSeqLen = 16;
        int batchSize = 32;
        double dropout = 0.1;

        var data = DataLoader.LoadData (
            maxSeqLen,
            out var vocabSize,
            out var vocabulary,
            sourceFile);

        var model = new Model (vocabSize, embeddingSize, numHeads, numLayers, maxSeqLen, PositionalEncodingType.Sinusoidal);
        model.DropoutRate = dropout;
        var optimizer = new Optimizer (learningRate: 0.0005);

        Trainer.train (
            model,
            optimizer,
            batchSize,
            data,
            vocabulary);
    }
}
