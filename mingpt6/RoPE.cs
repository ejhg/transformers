namespace mingpt6;

public class RoPE
{
    private int hiddenSize;

    public RoPE (int hiddenSize, int maxPosition) {
        this.hiddenSize = hiddenSize;
    }

    public double[] ApplyRoPE (double[] x, int position) {
        double[] result = new double[hiddenSize];
        for (int i = 0; i < hiddenSize / 2; i++) {
            double theta = position / Math.Pow (10000, 2.0 * i / hiddenSize);
            double cosTheta = Math.Cos (theta);
            double sinTheta = Math.Sin (theta);

            result[2 * i] = x[2 * i] * cosTheta - x[2 * i + 1] * sinTheta;
            result[2 * i + 1] = x[2 * i] * sinTheta + x[2 * i + 1] * cosTheta;
        }

        return result;
    }
}
