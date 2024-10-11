namespace mingpt6;

public class Optimizer
{
    private double learningRate;

    public Optimizer (double learningRate) {
        this.learningRate = learningRate;
    }

    public void Update (Matrix param, Matrix grad) {
        for (int i = 0; i < param.Rows; i++)
        for (int j = 0; j < param.Cols; j++)
            param.Data[i][j] -= learningRate * grad.Data[i][j];
    }

    public void Update (double[] param, double[] grad) {
        for (int i = 0; i < param.Length; i++)
            param[i] -= learningRate * grad[i];
    }
}
