namespace mingpt.cs;

public class Matrix
{
    public int Rows, Cols;
    public double[,] Data;

    public Matrix (int rows, int cols) {
        Rows = rows;
        Cols = cols;
        Data = new double[rows, cols];
    }

    public Matrix (double[,] data) {
        Rows = data.GetLength (0);
        Cols = data.GetLength (1);
        this.Data = data;
    }

    public Matrix (double[][] data) {
        throw new NotImplementedException ();
    }

    public static Matrix Random (int rows, int cols) {
        var m = new Matrix (rows, cols);
        var rand = new Random ();
        double std = 1.0 / Math.Sqrt (cols);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.Data[i, j] = rand.NextDouble () * 2 * std - std;
        return m;
    }

    /// <summary>
    /// Initialize weights with normal distribution (mean=0, std=0.02) like GPT-2
    /// </summary>
    public static Matrix RandomNormal(int rows, int cols, double mean = 0.0, double std = 0.02)
    {
        var m = new Matrix(rows, cols);
        var rand = new Random();
        
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Box-Muller transform for normal distribution
                double u1 = 1.0 - rand.NextDouble(); // uniform(0,1] 
                double u2 = 1.0 - rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                m.Data[i, j] = mean + std * randStdNormal;
            }
        }
        return m;
    }

    /// <summary>
    /// Initialize weights with zeros
    /// </summary>
    public static Matrix Zeros(int rows, int cols)
    {
        return new Matrix(rows, cols); // Already initialized to zeros
    }

    /// <summary>
    /// Initialize weights with ones
    /// </summary>
    public static Matrix Ones(int rows, int cols)
    {
        var m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                m.Data[i, j] = 1.0;
            }
        }
        return m;
    }

    public Matrix Transpose () {
        var result = new Matrix (Cols, Rows);
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            result.Data[j, i] = Data[i, j];
        return result;
    }

    public static Matrix operator + (Matrix a, Matrix b) {
        var result = new Matrix (a.Rows, a.Cols);

        if (b.Data.GetLength (0) == 1) {
            // broadcast
            for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Cols; j++)
                result.Data[i, j] = a.Data[i, j] + b.Data[0, j];
        } else {
            for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Cols; j++)
                result.Data[i, j] = a.Data[i, j] + b.Data[i, j];
        }

        return result;
    }

    public static Matrix operator - (Matrix a, Matrix b) {
        var result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i, j] = a.Data[i, j] - b.Data[i, j];
        return result;
    }

    public static Matrix operator * (Matrix a, Matrix b) {
        var result = new Matrix (a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < b.Cols; j++)
        for (int k = 0; k < a.Cols; k++)
            result.Data[i, j] += a.Data[i, k] * b.Data[k, j];
        return result;
    }

    public static Matrix operator * (double scalar, Matrix a) {
        var result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i, j] = scalar * a.Data[i, j];
        return result;
    }

    public static Matrix operator / (Matrix a, double scalar) {
        var result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i, j] = a.Data[i, j] / scalar;
        return result;
    }

    public Matrix SumRows () {
        var result = new Matrix (1, Cols);
        for (int j = 0; j < Cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < Rows; i++)
                sum += Data[i, j];
            result.Data[0, j] = sum;
        }

        return result;
    }

    public void Clear () {
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            Data[i, j] = 0.0;
    }

    public static Matrix operator + (Matrix a) {
        return a;
    }

    public static Matrix operator - (Matrix a) {
        var result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i, j] = -a.Data[i, j];
        return result;
    }

    public static Matrix operator + (Matrix a, double scalar) {
        var result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i, j] = a.Data[i, j] + scalar;
        return result;
    }

    public static Matrix operator + (double scalar, Matrix a) {
        return a + scalar;
    }
}
