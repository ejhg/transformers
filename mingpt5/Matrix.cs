namespace mingpt5;

public class Matrix
{
    public int Rows;
    public int Cols;
    public double[][] Data;

    public Matrix (int rows, int cols) {
        Rows = rows;
        Cols = cols;
        Data = new double[rows][];
        for (int i = 0; i < rows; i++) {
            Data[i] = new double[cols];
        }
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

    public static Matrix operator + (Matrix a, Matrix b) {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new Exception ("Matrix dimensions do not match");

        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] + b.Data[i][j];
        return result;
    }

    public static Matrix operator - (Matrix a, Matrix b) {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new Exception ("Matrix dimensions do not match");

        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] - b.Data[i][j];
        return result;
    }

    public static Matrix operator * (double scalar, Matrix a) {
        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = scalar * a.Data[i][j];
        return result;
    }

    public Matrix Transpose () {
        Matrix result = new Matrix (Cols, Rows);
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            result.Data[j][i] = Data[i][j];
        return result;
    }

    public Vector Multiply (Vector v) {
        if (Cols != v.Size)
            throw new Exception ("Matrix and vector dimensions are not compatible");

        Vector result = new Vector (Rows);
        for (int i = 0; i < Rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < Cols; j++) {
                sum += Data[i][j] * v.Data[j];
            }

            result.Data[i] = sum;
        }

        return result;
    }

    public static Matrix OuterProduct (Vector a, Vector b) {
        Matrix result = new Matrix (a.Size, b.Size);
        for (int i = 0; i < a.Size; i++)
        for (int j = 0; j < b.Size; j++)
            result.Data[i][j] = a.Data[i] * b.Data[j];
        return result;
    }
}
