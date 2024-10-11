namespace mingpt4;

class Vector
{
    public int Size { get; set; }
    public double[] Data { get; set; }

    public Vector (int size) {
        Size = size;
        Data = new double[size];
    }
}
