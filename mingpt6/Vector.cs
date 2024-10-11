namespace mingpt6;

public class Vector
{
    public double[] Data;
    public int Length;

    public Vector (int length) {
        Length = length;
        Data = new double[length];
    }

    public Vector (double[] data) {
        Data = data;
        Length = data.Length;
    }

    public static Vector operator + (Vector a, Vector b) {
        Vector result = new Vector (a.Length);
        for (int i = 0; i < a.Length; i++)
            result.Data[i] = a.Data[i] + b.Data[i];
        return result;
    }

    public static Vector operator - (Vector a, Vector b) {
        Vector result = new Vector (a.Length);
        for (int i = 0; i < a.Length; i++)
            result.Data[i] = a.Data[i] - b.Data[i];
        return result;
    }

    public static Vector operator * (double scalar, Vector a) {
        Vector result = new Vector (a.Length);
        for (int i = 0; i < a.Length; i++)
            result.Data[i] = scalar * a.Data[i];
        return result;
    }

    public static double Dot (Vector a, Vector b) {
        double result = 0;
        for (int i = 0; i < a.Length; i++)
            result += a.Data[i] * b.Data[i];
        return result;
    }
}
