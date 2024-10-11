namespace mingpt7;

public class LayerNorm
{
    public int size;
    public double[] gamma;
    public double[] beta;
    public double[] x_centered;
    public double[] std_inv;
    public double[] x_hat;
    public double[] gradGamma;
    public double[] gradBeta;

    public LayerNorm (int size) {
        this.size = size;
        gamma = new double[size];
        beta = new double[size];
        gradGamma = new double[size];
        gradBeta = new double[size];
        for (int i = 0; i < size; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    public double[] Forward (double[] x) {
        double mean = 0.0;
        for (int i = 0; i < size; i++)
            mean += x[i];
        mean /= size;

        double variance = 0.0;
        for (int i = 0; i < size; i++)
            variance += (x[i] - mean) * (x[i] - mean);
        variance /= size;

        std_inv = new double[size];
        x_centered = new double[size];
        x_hat = new double[size];
        double stdDev = Math.Sqrt (variance + 1e-5);

        for (int i = 0; i < size; i++) {
            x_centered[i] = x[i] - mean;
            std_inv[i] = 1.0 / stdDev;
            x_hat[i] = x_centered[i] * std_inv[i];
        }

        double[] outp = new double[size];
        for (int i = 0; i < size; i++)
            outp[i] = gamma[i] * x_hat[i] + beta[i];
        return outp;
    }

    public double[] Backward (double[] gradOutput) {
        // Compute gradients
        double[] gradXHat = new double[size];
        for (int i = 0; i < size; i++) {
            gradGamma[i] += gradOutput[i] * x_hat[i];
            gradBeta[i] += gradOutput[i];
            gradXHat[i] = gradOutput[i] * gamma[i];
        }

        double[] gradX = new double[size];
        double meanGradXHat = 0.0;
        double meanXCentered = 0.0;

        for (int i = 0; i < size; i++) {
            meanGradXHat += gradXHat[i];
            meanXCentered += x_centered[i];
        }

        meanGradXHat /= size;
        meanXCentered /= size;

        for (int i = 0; i < size; i++) {
            gradX[i] = (gradXHat[i] - meanGradXHat - x_centered[i] * meanGradXHat / (std_inv[i] * std_inv[i] * size)) * std_inv[i];
        }

        return gradX;
    }

    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < size; i++) {
            gamma[i] -= learningRate * gradGamma[i];
            beta[i] -= learningRate * gradBeta[i];
            gradGamma[i] = 0.0;
            gradBeta[i] = 0.0;
        }
    }
}
