namespace mingpt7;

public class FeedForward
{
    public int embeddingDim;
    public int hiddenDim;
    public double[,] W1; // [hiddenDim, embeddingDim]
    public double[] b1; // [hiddenDim]
    public double[,] W2; // [embeddingDim, hiddenDim]
    public double[] b2; // [embeddingDim]

    // Intermediate variables for backprop
    double[] x_input;
    double[] hidden;
    double[] hiddenActivation;

    // Gradients
    double[,] dW1;
    double[] db1;
    double[,] dW2;
    double[] db2;

    public FeedForward (int embeddingDim, int hiddenDim) {
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
        W1 = new double[hiddenDim, embeddingDim];
        b1 = new double[hiddenDim];
        W2 = new double[embeddingDim, hiddenDim];
        b2 = new double[embeddingDim];

        dW1 = new double[hiddenDim, embeddingDim];
        db1 = new double[hiddenDim];
        dW2 = new double[embeddingDim, hiddenDim];
        db2 = new double[embeddingDim];

        Random rand = new Random ();
        InitializeMatrix (W1, rand);
        InitializeVector (b1, rand);
        InitializeMatrix (W2, rand);
        InitializeVector (b2, rand);
    }

    void InitializeMatrix (double[,] matrix, Random rand) {
        int rows = matrix.GetLength (0);
        int cols = matrix.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 0.01;
    }

    void InitializeVector (double[] vector, Random rand) {
        for (int i = 0; i < vector.Length; i++)
            vector[i] = 0.0;
    }

    public double[] Forward (double[] x) {
        x_input = x;
        hidden = math.MatVecMul (W1, x);
        hidden = math.Add (hidden, b1);
        // Apply activation function, e.g., ReLU
        hiddenActivation = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++)
            hiddenActivation[i] = Math.Max (0, hidden[i]);

        double[] outp = math.MatVecMul (W2, hiddenActivation);
        outp = math.Add (outp, b2);
        return outp;
    }

    public double[] Backward (double[] gradOutput) {
        // Gradient w.r.t W2, b2
        for (int i = 0; i < embeddingDim; i++)
        for (int j = 0; j < hiddenDim; j++)
            dW2[i, j] += gradOutput[i] * hiddenActivation[j];

        for (int i = 0; i < embeddingDim; i++)
            db2[i] += gradOutput[i];

        // Gradient w.r.t hidden activation
        double[] gradHiddenActivation = new double[hiddenDim];
        for (int j = 0; j < hiddenDim; j++)
        for (int i = 0; i < embeddingDim; i++)
            gradHiddenActivation[j] += W2[i, j] * gradOutput[i];

        // Gradient w.r.t hidden pre-activation
        double[] gradHidden = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++)
            gradHidden[i] = hidden[i] > 0 ? gradHiddenActivation[i] : 0.0;

        // Gradient w.r.t W1, b1
        for (int i = 0; i < hiddenDim; i++)
        for (int j = 0; j < embeddingDim; j++)
            dW1[i, j] += gradHidden[i] * x_input[j];

        for (int i = 0; i < hiddenDim; i++)
            db1[i] += gradHidden[i];

        // Gradient w.r.t input x
        double[] gradInput = new double[embeddingDim];
        for (int j = 0; j < embeddingDim; j++)
        for (int i = 0; i < hiddenDim; i++)
            gradInput[j] += W1[i, j] * gradHidden[i];

        return gradInput;
    }

    public void UpdateParameters (double learningRate) {
        // Update W1, b1
        for (int i = 0; i < hiddenDim; i++)
        for (int j = 0; j < embeddingDim; j++) {
            W1[i, j] -= learningRate * dW1[i, j];
            dW1[i, j] = 0.0;
        }

        for (int i = 0; i < hiddenDim; i++) {
            b1[i] -= learningRate * db1[i];
            db1[i] = 0.0;
        }

        // Update W2, b2
        for (int i = 0; i < embeddingDim; i++)
        for (int j = 0; j < hiddenDim; j++) {
            W2[i, j] -= learningRate * dW2[i, j];
            dW2[i, j] = 0.0;
        }

        for (int i = 0; i < embeddingDim; i++) {
            b2[i] -= learningRate * db2[i];
            db2[i] = 0.0;
        }
    }
}
