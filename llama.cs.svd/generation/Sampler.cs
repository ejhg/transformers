namespace llama.cs.svd;

/**
 * The sampler takes in logits and returns a sampled token
 */
public class Sampler
{
    int vocab_size;

    float temperature;

    float topp;

    Random rng;

    public Sampler (int vocab_size, float temperature, float topp, int? rng_seed) {
        this.vocab_size = vocab_size;
        this.temperature = temperature;
        this.topp = topp;
        rng = rng_seed == null
            ? new Random ()
            : new Random (rng_seed.Value);
    }

    int SampleArgMax (float[] probabilities) {
        int max_i = 0;
        float max_p = probabilities[0];
        for (int i = 1; i < probabilities.Length; i++) {
            if (probabilities[i] > max_p) {
                max_p = probabilities[i];
                max_i = i;
            }
        }

        return max_i;
    }

    int SampleMult (float[] probabilities) {
        float coin = (float)rng.NextDouble ();
        float cdf = 0.0f;
        for (int i = 0; i < probabilities.Length; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }

        return probabilities.Length - 1;
    }

    int SampleTopP (float[] probabilities) {
        float coin = (float)rng.NextDouble ();
        int n0 = 0;
        float cutoff = (1.0f - topp) / (probabilities.Length - 1);

        var probindex = new (float prob, int index)[vocab_size];

        for (int i = 0; i < probabilities.Length; i++) {
            if (probabilities[i] >= cutoff) {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                n0++;
            }
        }

        var sorted = probindex
            .Take (n0)
            .OrderByDescending (_ => _.prob)
            .ToArray ();

        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1;
        for (int i = 0; i < n0; i++) {
            cumulative_prob += sorted[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;
            }
        }

        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += sorted[i].prob;
            if (r < cdf) {
                return sorted[i].index;
            }
        }

        return sorted[last_idx].index;
    }

    public int Sample (float[] logits) {
        if (temperature == 0.0f) {
            return SampleArgMax (logits);
        }

        for (int i = 0; i < logits.Length; i++) {
            logits[i] /= temperature;
        }

        math.Softmax (logits, logits.Length);

        if (topp <= 0 || topp >= 1) {
            return SampleMult (logits);
        }

        return SampleTopP (logits);
    }
}
