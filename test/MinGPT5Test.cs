namespace mingpt5;

class MinGPT5Test
{
    public static void run () {
        int embeddingDim = 96;
        int numHeads = 6;
        int hiddenDim = 256;
        int numLayers = 4;

        var data = LoadData (
            sequenceLength: 8,
            out var vocabSize,
            out var vocabulary);

        TransformerModel model = new TransformerModel (vocabSize, embeddingDim, numHeads, hiddenDim, numLayers);
        Trainer trainer = new Trainer (model, 0.0005);

        trainer.Train (data, epochs: 100);

        int[] testSequence = new int[] {
            1,
            2,
            3
        };
        int predictedToken = trainer.PredictNextToken (testSequence);
        Console.WriteLine ($"Predicted next token: {predictedToken}");
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
