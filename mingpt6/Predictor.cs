namespace mingpt6;

public class Predictor
{
    private Transformer model;

    public Predictor (Transformer model) {
        this.model = model;
    }

    public int PredictNextToken (int[] inputIds) {
        double[] probabilities = model.Forward (inputIds);
        int predictedId = ArgMax (probabilities);
        return predictedId;
    }

    private int ArgMax (double[] array) {
        int index = 0;
        double max = array[0];
        for (int i = 1; i < array.Length; i++)
            if (array[i] > max) {
                max = array[i];
                index = i;
            }

        return index;
    }
}
