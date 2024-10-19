namespace llama;

public class LayerNorm
{
    public double[] Gamma;
    public double[] Beta;
    public int Size;

    public double[] dGamma;
    public double[] dBeta;

    // For Adam optimizer
    public double[] mGamma;
    public double[] vGamma;
    public double[] mBeta;
    public double[] vBeta;

    public LayerNorm (int size) {
        Size = size;
        Gamma = new double[size];
        Beta = new double[size];
        dGamma = new double[size];
        dBeta = new double[size];

        mGamma = new double[size];
        vGamma = new double[size];
        mBeta = new double[size];
        vBeta = new double[size];

        for (int i = 0; i < size; i++) {
            Gamma[i] = 1.0;
            Beta[i] = 0.0;
        }
    }

    public (double[] output, LayerNormCache cache) Forward (double[] x) {
        double mean = x.Average ();
        double variance = x.Select (val => Math.Pow (val - mean, 2)).Average ();
        double[] normalized = x.Select (val => (val - mean) / Math.Sqrt (variance + 1e-5)).ToArray ();
        double[] output = new double[Size];
        for (int i = 0; i < Size; i++)
            output[i] = Gamma[i] * normalized[i] + Beta[i];

        var cache = new LayerNormCache {
            x_input = x,
            mean = mean,
            variance = variance,
            normalized = normalized
        };

        return (output, cache);
    }

    public double[] Backward (double[] gradOutput, LayerNormCache cache) {
        double[] dxhat = new double[Size];
        for (int i = 0; i < Size; i++) {
            dGamma[i] += gradOutput[i] * cache.normalized[i];
            dBeta[i] += gradOutput[i];
            dxhat[i] = gradOutput[i] * Gamma[i];
        }

        double stdInv = 1.0 / Math.Sqrt (cache.variance + 1e-5);
        double[] dx = new double[Size];
        double dvar = -0.5 * stdInv * stdInv * stdInv * dxhat.Select ((dxh, i) => (cache.x_input[i] - cache.mean) * dxh).Sum ();
        double dmean = -stdInv * dxhat.Sum () + dvar * (-2.0 / Size) * (cache.x_input.Sum () - Size * cache.mean);
        for (int i = 0; i < Size; i++)
            dx[i] = stdInv * dxhat[i] + dvar * 2.0 * (cache.x_input[i] - cache.mean) / Size + dmean / Size;
        return dx;
    }
}

public class LayerNormCache
{
    public double[] x_input;
    public double mean;
    public double variance;
    public double[] normalized;
}
