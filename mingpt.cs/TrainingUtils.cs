namespace mingpt.cs;

public static class TrainingUtils
{
    public static double ComputePerplexity (double averageLoss) {
        return Math.Exp (averageLoss);
    }

    public static void SetTrainingMode (Model model, bool training) {
        model.Training = training;
    }

    public static void SetEvalMode (Model model) {
        SetTrainingMode (model, false);
    }

    public static int ArgMax (double[] logits) {
        int maxIndex = 0;
        double maxValue = logits[0];
        for (int i = 1; i < logits.Length; i++) {
            if (logits[i] > maxValue) {
                maxValue = logits[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static double[] ExtractLastTokenLogits (Matrix logits) {
        int seqLen = logits.Rows;
        int vocabSize = logits.Cols;
        double[] lastLogits = new double[vocabSize];
        
        for (int i = 0; i < vocabSize; i++) {
            lastLogits[i] = logits.Data[seqLen - 1, i];
        }
        
        return lastLogits;
    }

    public static double[] Softmax (double[] logits) {
        double max = double.NegativeInfinity;
        foreach (var val in logits) {
            if (val > max) max = val;
        }
        
        double sum = 0;
        double[] exp = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++) {
            exp[i] = Math.Exp (logits[i] - max);
            sum += exp[i];
        }
        
        for (int i = 0; i < logits.Length; i++) {
            exp[i] /= sum;
        }
        
        return exp;
    }
}