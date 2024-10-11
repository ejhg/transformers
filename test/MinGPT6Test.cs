using System;

namespace mingpt6;

public class MinGPT6Test
{
    public static void run () {
        int hiddenSize = 96;
        int numHeads = 8;
        int maxPosition = 512;

        var data = LoadData (sequenceLength: 8, out var vocabSize, out var dictionary);

        Transformer model = new Transformer (vocabSize, hiddenSize, numHeads, maxPosition);
        Optimizer optimizer = new Optimizer (learningRate: 0.0005);
        TrainingLoop trainer = new TrainingLoop (model, optimizer, vocabSize);

        trainer.Train (data, epochs: 100);

        // Prediction
        Predictor predictor = new Predictor (model);
        int[] inputIds = new int[] {
            1,
            2,
            3
        };
        int nextTokenId = predictor.PredictNextToken (inputIds);
        Console.WriteLine ($"Predicted next token ID: {nextTokenId}");
    }

    static Func<(int[], int)> LoadData (int sequenceLength, out int vocabularySize, out char[] vocabulary) {
        var text = File.ReadAllText ("resources/tinyshakespeare.txt");
        var sourceCharacters = text.ToArray ();

        vocabulary = sourceCharacters
            .Distinct ()
            .Order ()
            .ToArray ();
        vocabularySize = vocabulary.Length;

        Console.WriteLine ($"vocabulary: {string.Join ("", vocabulary)}");

        var charToIndex = vocabulary
            .Select ((c, index) => (c, index))
            .ToDictionary ();

        var data = sourceCharacters.Select (c => charToIndex[c]).ToArray ();

        var rnd = new Random ();

        return () => {
            var sample = data
                .Skip (rnd.Next (0, data.Length - sequenceLength - 1))
                .Take (sequenceLength + 1)
                .ToArray ();

            return (sample.Take (sequenceLength).ToArray (), sample[^1]);
        };
    }
}
