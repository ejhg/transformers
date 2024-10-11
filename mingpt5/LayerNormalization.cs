namespace mingpt5;

public class LayerNormalization
{
    public int FeatureSize;
    public Vector Gamma;
    public Vector Beta;
    public Vector Input;
    public Vector Normalized;
    public double Mean;
    public double Variance;

    public LayerNormalization (int featureSize) {
        FeatureSize = featureSize;
        Gamma = new Vector (featureSize);
        Beta = new Vector (featureSize);
        InitializeParameters ();
    }

    private void InitializeParameters () {
        for (int i = 0; i < FeatureSize; i++) {
            Gamma.Data[i] = 1.0;
            Beta.Data[i] = 0.0;
        }
    }

    public Vector Forward (Vector input) {
        Input = input.Clone ();
        Mean = 0.0;
        Variance = 0.0;

        for (int i = 0; i < FeatureSize; i++) {
            Mean += Input.Data[i];
        }

        Mean /= FeatureSize;

        for (int i = 0; i < FeatureSize; i++) {
            Variance += Math.Pow (Input.Data[i] - Mean, 2);
        }

        Variance /= FeatureSize;

        Normalized = new Vector (FeatureSize);
        for (int i = 0; i < FeatureSize; i++) {
            Normalized.Data[i] = (Input.Data[i] - Mean) / Math.Sqrt (Variance + 1e-6);
            Normalized.Data[i] = Gamma.Data[i] * Normalized.Data[i] + Beta.Data[i];
        }

        return Normalized;
    }

    public Vector Backward (Vector dout) {
        Vector dGamma = new Vector (FeatureSize);
        Vector dBeta = new Vector (FeatureSize);
        Vector dx = new Vector (FeatureSize);

        double invStd = 1.0 / Math.Sqrt (Variance + 1e-6);
        Vector xHat = new Vector (FeatureSize);
        for (int i = 0; i < FeatureSize; i++) {
            xHat.Data[i] = (Input.Data[i] - Mean) * invStd;
        }

        for (int i = 0; i < FeatureSize; i++) {
            dGamma.Data[i] += dout.Data[i] * xHat.Data[i];
            dBeta.Data[i] += dout.Data[i];
        }

        for (int i = 0; i < FeatureSize; i++) {
            double dXhat = dout.Data[i] * Gamma.Data[i];
            double dVar = -0.5 * dXhat * (Input.Data[i] - Mean) * Math.Pow (Variance + 1e-6, -1.5);
            double dMean = -dXhat * invStd;
            dx.Data[i] += dXhat * invStd + dVar * 2.0 * (Input.Data[i] - Mean) / FeatureSize + dMean / FeatureSize;
        }

        // Update parameters
        for (int i = 0; i < FeatureSize; i++) {
            Gamma.Data[i] -= dout.Data[i] * xHat.Data[i];
            Beta.Data[i] -= dout.Data[i];
        }

        return dx;
    }
}
