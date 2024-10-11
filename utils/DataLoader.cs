namespace transformers.utils;

static class DataLoader
{
    public static Func<(int[], int)> LoadData (int sequenceLength, out int vocabularySize, out char[] vocabulary) {
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
