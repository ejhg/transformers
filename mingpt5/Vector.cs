namespace mingpt5;

public class Vector
{
    public int Size;
    public double[] Data;

    public Vector (int size) {
        Size = size;
        Data = new double[size];
    }

    public Vector (double[] data) {
        Size = data.Length;
        Data = new double[Size];
        Array.Copy (data, Data, Size);
    }

    public static Vector operator + (Vector a, Vector b) {
        if (a.Size != b.Size)
            throw new Exception ("Vector sizes do not match");

        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++) {
            result.Data[i] = a.Data[i] + b.Data[i];
        }

        return result;
    }

    public static Vector operator - (Vector a, Vector b) {
        if (a.Size != b.Size)
            throw new Exception ("Vector sizes do not match");

        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++) {
            result.Data[i] = a.Data[i] - b.Data[i];
        }

        return result;
    }

    public static Vector operator * (double scalar, Vector a) {
        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++) {
            result.Data[i] = scalar * a.Data[i];
        }

        return result;
    }

    public void ApplyFunction (Func<double, double> func) {
        for (int i = 0; i < Size; i++) {
            Data[i] = func (Data[i]);
        }
    }

    public double Dot (Vector other) {
        if (Size != other.Size)
            throw new Exception ("Vector sizes do not match");

        double sum = 0.0;
        for (int i = 0; i < Size; i++) {
            sum += Data[i] * other.Data[i];
        }

        return sum;
    }

    public Vector Clone () {
        return new Vector (Data);
    }
}
