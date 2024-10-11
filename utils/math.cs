namespace transformers.utils;

public static class math
{
    // Vector addition
    public static double[] Add (double[] a, double[] b) {
        double[] result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] + b[i];
        return result;
    }

    // Scalar multiplication
    public static double[] Multiply (double[] a, double scalar) {
        double[] result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] * scalar;
        return result;
    }

    // Dot product
    public static double Dot (double[] a, double[] b) {
        double result = 0.0;
        for (int i = 0; i < a.Length; i++)
            result += a[i] * b[i];
        return result;
    }

    // Matrix-vector multiplication
    public static double[] MatVecMul (double[,] matrix, double[] vector) {
        int rows = matrix.GetLength (0);
        int cols = matrix.GetLength (1);
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++)
                sum += matrix[i, j] * vector[j];
            result[i] = sum;
        }

        return result;
    }

    // Softmax function
    public static double[] Softmax (double[] x) {
        double max = double.MinValue;
        for (int i = 0; i < x.Length; i++)
            if (x[i] > max)
                max = x[i];
        double sum = 0.0;
        double[] expX = new double[x.Length];
        for (int i = 0; i < x.Length; i++) {
            expX[i] = Math.Exp (x[i] - max);
            sum += expX[i];
        }

        for (int i = 0; i < x.Length; i++)
            expX[i] /= sum;
        return expX;
    }

    // Cross-entropy loss
    public static double CrossEntropyLoss (double[] probs, int targetIndex) {
        return -Math.Log (probs[targetIndex] + 1e-12);
    }

    // Apply rotational positional encoding (ROPE)
    public static double[] ApplyROPE (double[] x, int position) {
        int D = x.Length;
        double[] x_out = new double[D];
        for (int i = 0; i < D / 2; i++) {
            double theta = position / Math.Pow (10000, 2.0 * i / D);
            double cosTheta = Math.Cos (theta);
            double sinTheta = Math.Sin (theta);
            int idx = 2 * i;
            x_out[idx] = x[idx] * cosTheta - x[idx + 1] * sinTheta;
            x_out[idx + 1] = x[idx] * sinTheta + x[idx + 1] * cosTheta;
        }

        if (D % 2 == 1)
            x_out[D - 1] = x[D - 1];
        return x_out;
    }
}
