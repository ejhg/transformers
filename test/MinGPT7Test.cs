using transformers.utils;

namespace mingpt7;

public class MinGPT7Test
{
    public static void run () {
        int embeddingDim = 64;
        int numHeads = 4;
        int numLayers = 6;
        int hiddenDim = 128;
        double learningRate = 0.001;
        int numEpochs = 100000;

        var data = DataLoader.LoadData (sequenceLength: 8, out var vocabSize, out var dictionary);

        Transformer model = new Transformer (vocabSize, embeddingDim, numHeads, numLayers, hiddenDim);
        Trainer trainer = new Trainer (model, learningRate);

        trainer.Train (data, numEpochs);

        // int nextToken = trainer.PredictNextToken (tokenIndices);
        // Console.WriteLine ($"Next token predicted: {nextToken}");
    }
}
