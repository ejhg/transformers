namespace mingpt4;

class Vector
{
    public int Size { get; set; }
    public double[] Data { get; set; }

    public Vector (int size) {
        Size = size;
        Data = new double[size];
    }

    public Vector (double[] data) {
        Size = data.Length;
        Data = new double[Size];
        Array.Copy (data, Data, Size);
    }

    public static Vector Add (Vector a, Vector b) {
        if (a.Size != b.Size)
            throw new ArgumentException ("Vectors must be the same size.");

        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] + b.Data[i];
        return result;
    }

    public static Vector Subtract (Vector a, Vector b) {
        if (a.Size != b.Size)
            throw new ArgumentException ("Vectors must be the same size.");

        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] - b.Data[i];
        return result;
    }

    public static Vector Multiply (Vector a, double scalar) {
        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] * scalar;
        return result;
    }

    public static double Dot (Vector a, Vector b) {
        if (a.Size != b.Size)
            throw new ArgumentException ("Vectors must be the same size.");

        double result = 0;
        for (int i = 0; i < a.Size; i++)
            result += a.Data[i] * b.Data[i];
        return result;
    }

    public Vector Clone () {
        return new Vector (Data);
    }
}
