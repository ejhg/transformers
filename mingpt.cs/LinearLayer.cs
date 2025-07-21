namespace mingpt.cs;

public class LinearLayer
{
    public Matrix Weights;
    public Matrix Bias;
    public Matrix GradWeights;
    public Matrix GradBias;
    private Matrix Input;

    public LinearLayer (int inputSize, int outputSize) {
        Weights = Matrix.Random (inputSize, outputSize);
        Bias = new Matrix (1, outputSize);
        GradWeights = new Matrix (inputSize, outputSize);
        GradBias = new Matrix (1, outputSize);
    }

    public Matrix Forward (Matrix input) {
        Input = input;
        return (input * Weights) + Bias;
    }

    public Matrix Backward (Matrix dOutput) {
        GradWeights += Input.Transpose () * dOutput;
        GradBias += dOutput.SumRows ();
        return dOutput * Weights.Transpose ();
    }

    public void UpdateParameters (double LearningRate) {
        Weights -= LearningRate * GradWeights;
        Bias -= LearningRate * GradBias;
        GradWeights.Clear ();
        GradBias.Clear ();
    }
}
