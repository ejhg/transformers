namespace mingpt6;

public class Matrix
{
    public double[][] Data;
    public int Rows;
    public int Cols;

    public Matrix (int rows, int cols) {
        Rows = rows;
        Cols = cols;
        Data = new double[rows][];
        for (int i = 0; i < rows; i++)
            Data[i] = new double[cols];
    }

    public Matrix (double[][] data) {
        Data = data;
        Rows = data.Length;
        Cols = data[0].Length;
    }

    public static Matrix operator + (Matrix a, Matrix b) {
        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] + b.Data[i][j];
        return result;
    }

    public static Matrix operator - (Matrix a, Matrix b) {
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

    public static Matrix Multiply (Matrix a, Matrix b) {
        if (a.Cols != b.Rows)
            throw new Exception ("Matrix dimensions do not match for multiplication.");

        Matrix result = new Matrix (a.Rows, b.Cols);

        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < b.Cols; j++)
        for (int k = 0; k < a.Cols; k++)
            result.Data[i][j] += a.Data[i][k] * b.Data[k][j];

        return result;
    }

    public Matrix Transpose () {
        Matrix result = new Matrix (Cols, Rows);
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            result.Data[j][i] = Data[i][j];
        return result;
    }

    public Matrix Clone () {
        double[][] newData = new double[Rows][];
        for (int i = 0; i < Rows; i++)
            newData[i] = (double[])Data[i].Clone ();
        return new Matrix (newData);
    }
}
