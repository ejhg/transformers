namespace mingpt.cs;

public class LinearLayer
{
    public Matrix Weights;
    public Matrix Bias;
    public Matrix GradWeights;
    public Matrix GradBias;
    private Matrix Input;

    public LinearLayer (int inputSize, int outputSize, bool useBias = true) {
        Weights = Matrix.RandomNormal (inputSize, outputSize);
        Bias = useBias ? Matrix.Zeros(1, outputSize) : null;
        GradWeights = new Matrix (inputSize, outputSize);
        GradBias = useBias ? new Matrix (1, outputSize) : null;
    }

    public Matrix Forward (Matrix input) {
        Input = input;
        return Bias != null ? (input * Weights) + Bias : input * Weights;
    }

    public Matrix Backward (Matrix dOutput) {
        GradWeights += Input.Transpose () * dOutput;
        if (GradBias != null)
            GradBias += dOutput.SumRows ();
        return dOutput * Weights.Transpose ();
    }

    public void UpdateParameters (double LearningRate) {
        Weights -= LearningRate * GradWeights;
        if (Bias != null)
            Bias -= LearningRate * GradBias;
        GradWeights.Clear ();
        if (GradBias != null)
            GradBias.Clear ();
    }
}
