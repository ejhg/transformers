namespace transformers.utils;

static class DataLoader
{
    public static Func<(int[], int[])> LoadData (int sequenceLength, out int vocabularySize, out char[] vocabulary) {
        var text = File.ReadAllText ("resources/tinyshakespeare.txt");
        var sourceCharacters = text.ToArray ();

        var _vocabulary = sourceCharacters
            .Distinct ()
            .Order ()
            .ToArray ();
        vocabulary = _vocabulary;
        vocabularySize = vocabulary.Length;

        Console.WriteLine ($"vocabulary: {string.Join ("", vocabulary)}");

        var data = encode (text, vocabulary);

        var rnd = new Random ();

        return () => {
            var sample = data
                .Skip (rnd.Next (0, data.Length - sequenceLength - 1))
                .Take (sequenceLength + 1)
                .ToArray ();

            return (sample.Take (sequenceLength).ToArray (), sample.Skip (1).Take (sequenceLength).ToArray ());
        };
    }

    public static int[] encode (string source, char[] vocabulary) {
        var charToIndex = vocabulary
            .Select ((c, index) => (c, index))
            .ToDictionary ();
        return source.Select (c => charToIndex[c]).ToArray ();
    }

    public static string decode (int[] tokens, char[] vocabulary) {
        return string.Join ("", tokens.Select (c => c >= vocabulary.Length ? '?' : vocabulary[c]).ToArray ());
    }
}
