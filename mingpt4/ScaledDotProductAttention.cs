using mingpt5;

namespace mingpt4;

class ScaledDotProductAttention
{
    public static Matrix ComputeAttention (Matrix Q, Matrix K, Matrix V) {
        Matrix K_T = K.Transpose ();
        Matrix scores = Matrix.Multiply (Q, K_T);

        double scale = Math.Sqrt (Q.Cols);
        for (int i = 0; i < scores.Rows; i++)
        for (int j = 0; j < scores.Cols; j++)
            scores.Data[i][j] /= scale;

        // Apply masking for autoregressive behavior
        for (int i = 0; i < scores.Rows; i++)
        for (int j = i + 1; j < scores.Cols; j++)
            scores.Data[i][j] = double.NegativeInfinity;

        // Apply softmax to scores
        Matrix attentionWeights = Softmax (scores);

        // Compute weighted sum of values
        Matrix output = Matrix.Multiply (attentionWeights, V);

        return output;
    }

    private static Matrix Softmax (Matrix input) {
        Matrix result = new Matrix (input.Rows, input.Cols);
        for (int i = 0; i < input.Rows; i++) {
            double max = double.MinValue;
            for (int j = 0; j < input.Cols; j++)
                if (input.Data[i][j] > max)
                    max = input.Data[i][j];

            double sum = 0;
            double[] expValues = new double[input.Cols];
            for (int j = 0; j < input.Cols; j++) {
                expValues[j] = Math.Exp (input.Data[i][j] - max);
                sum += expValues[j];
            }

            for (int j = 0; j < input.Cols; j++)
                result.Data[i][j] = expValues[j] / sum;
        }

        return result;
    }
}
