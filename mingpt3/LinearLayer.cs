namespace mingpt3;

public class LinearLayer
{
    public int InputSize, OutputSize;
    public Matrix Weights;
    public Matrix Bias;
    public Matrix GradWeights;
    public Matrix GradBias;
    private Matrix Input;

    public LinearLayer (int inputSize, int outputSize) {
        InputSize = inputSize;
        OutputSize = outputSize;
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
        var dInput = dOutput * Weights.Transpose ();
        return dInput;
    }
}
