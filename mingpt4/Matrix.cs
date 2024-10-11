namespace mingpt4;

class Matrix
{
    public int Rows { get; set; }
    public int Cols { get; set; }
    public double[][] Data { get; set; }

    public Matrix (int rows, int cols) {
        Rows = rows;
        Cols = cols;
        Data = new double[rows][];
        for (int i = 0; i < rows; i++)
            Data[i] = new double[cols];
    }

    public Matrix (double[][] data) {
        Rows = data.Length;
        Cols = data[0].Length;
        Data = new double[Rows][];
        for (int i = 0; i < Rows; i++) {
            Data[i] = new double[Cols];
            Array.Copy (data[i], Data[i], Cols);
        }
    }

    public static Matrix Multiply (Matrix a, Matrix b) {
        if (a.Cols != b.Rows)
            throw new ArgumentException ("Matrix dimensions are not suitable for multiplication.");

        Matrix result = new Matrix (a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++) {
            for (int j = 0; j < b.Cols; j++) {
                double sum = 0;
                for (int k = 0; k < a.Cols; k++)
                    sum += a.Data[i][k] * b.Data[k][j];
                result.Data[i][j] = sum;
            }
        }

        return result;
    }

    public static Vector Multiply (Matrix a, Vector b) {
        if (a.Cols != b.Size)
            throw new ArgumentException ("Matrix and vector dimensions are not suitable for multiplication.");

        Vector result = new Vector (a.Rows);
        for (int i = 0; i < a.Rows; i++) {
            double sum = 0;
            for (int j = 0; j < a.Cols; j++)
                sum += a.Data[i][j] * b.Data[j];
            result.Data[i] = sum;
        }

        return result;
    }

    public static Matrix Transpose (Matrix a) {
        Matrix result = new Matrix (a.Cols, a.Rows);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[j][i] = a.Data[i][j];
        return result;
    }

    public static Matrix Add (Matrix a, Matrix b) {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new ArgumentException ("Matrices must be the same size.");

        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] + b.Data[i][j];
        return result;
    }

    public static Matrix Subtract (Matrix a, Matrix b) {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new ArgumentException ("Matrices must be the same size.");

        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] - b.Data[i][j];
        return result;
    }

    public static Matrix Multiply (Matrix a, double scalar) {
        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] * scalar;
        return result;
    }

    public Matrix Clone () {
        return new Matrix (Data);
    }
}
