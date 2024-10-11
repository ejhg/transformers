namespace transformers.utils;

static class sampling
{
    public static int SampleFromDistribution (double[] probabilities) {
        var rand = new Random ();
        double cumulative = 0.0;
        double sample = rand.NextDouble ();
        for (int i = 0; i < probabilities.Length; i++) {
            cumulative += probabilities[i];
            if (sample < cumulative)
                return i;
        }

        return probabilities.Length - 1; // Fallback
    }

    public static int ArgMax (double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];
        for (int i = 1; i < array.Length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
