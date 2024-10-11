namespace mingpt6;

public class LayerNorm
{
    private int hiddenSize;
    private double[] gamma;
    private double[] beta;

    private double[] input;
    private double[] normalizedInput;
    private double mean;
    private double variance;
    private double epsilon = 1e-5;

    public LayerNorm (int hiddenSize) {
        this.hiddenSize = hiddenSize;
        gamma = new double[hiddenSize];
        beta = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
            gamma[i] = 1.0;
        // beta initialized to zeros
    }

    public double[] Forward (double[] input) {
        this.input = input;
        mean = 0;
        for (int i = 0; i < input.Length; i++)
            mean += input[i];
        mean /= input.Length;

        variance = 0;
        for (int i = 0; i < input.Length; i++)
            variance += (input[i] - mean) * (input[i] - mean);
        variance /= input.Length;

        normalizedInput = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            normalizedInput[i] = (input[i] - mean) / Math.Sqrt (variance + epsilon);

        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            output[i] = gamma[i] * normalizedInput[i] + beta[i];

        return output;
    }

    public double[] Backward (double[] gradOutput) {
        // Implement backpropagation for layer normalization
        // Compute gradients w.r.t gamma, beta, and input

        int N = input.Length;

        double[] gradGamma = new double[N];
        double[] gradBeta = new double[N];
        double[] gradInput = new double[N];

        for (int i = 0; i < N; i++) {
            gradGamma[i] = gradOutput[i] * normalizedInput[i];
            gradBeta[i] = gradOutput[i];
        }

        // Compute gradInput
        double[] dxhat = new double[N];
        for (int i = 0; i < N; i++)
            dxhat[i] = gradOutput[i] * gamma[i];

        double dvariance = 0;
        for (int i = 0; i < N; i++)
            dvariance += dxhat[i] * (input[i] - mean) * -0.5 * Math.Pow (variance + epsilon, -1.5);

        double dmean = 0;
        for (int i = 0; i < N; i++)
            dmean += dxhat[i] * -1 / Math.Sqrt (variance + epsilon);
        dmean += dvariance * -2 * mean / N;

        for (int i = 0; i < N; i++)
            gradInput[i] = dxhat[i] / Math.Sqrt (variance + epsilon) + dvariance * 2 * (input[i] - mean) / N + dmean / N;

        // Update gamma and beta parameters
        // Here, we should store gradGamma and gradBeta for parameter update in optimizer

        // For simplicity, let's assume we have functions to update gamma and beta

        return gradInput;
    }
}
