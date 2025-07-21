namespace mingpt.cs;

/// <summary>
/// Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
/// Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
/// 
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// </summary>
public static class NewGelu
{
    private static readonly double Sqrt2OverPi = Math.Sqrt(2.0 / Math.PI);
    private static readonly double Coeff = 0.044715;

    public static double[,] Forward(double[,] x)
    {
        int rows = x.GetLength(0);
        int cols = x.GetLength(1);
        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double val = x[i, j];
                double tanh_arg = Sqrt2OverPi * (val + Coeff * Math.Pow(val, 3.0));
                result[i, j] = 0.5 * val * (1.0 + Math.Tanh(tanh_arg));
            }
        }

        return result;
    }

    public static double[,] Backward(double[,] x, double[,] dOutput)
    {
        int rows = x.GetLength(0);
        int cols = x.GetLength(1);
        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double val = x[i, j];
                double tanh_arg = Sqrt2OverPi * (val + Coeff * Math.Pow(val, 3.0));
                double tanh_val = Math.Tanh(tanh_arg);
                double sech2_val = 1.0 - tanh_val * tanh_val; // sech^2 = 1 - tanh^2
                
                // Derivative of GELU
                double gelu_derivative = 0.5 * (1.0 + tanh_val) + 
                    0.5 * val * sech2_val * Sqrt2OverPi * (1.0 + 3.0 * Coeff * val * val);
                
                result[i, j] = dOutput[i, j] * gelu_derivative;
            }
        }

        return result;
    }
}