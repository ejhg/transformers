using mingpt5;

namespace mingpt4;

class CrossEntropyLoss
{
    public static double ComputeLoss (Matrix logits, int[] targets) {
        int seqLength = logits.Rows;
        int vocabSize = logits.Cols;
        double loss = 0;
        for (int i = 0; i < seqLength; i++) {
            double maxLogit = double.MinValue;
            for (int j = 0; j < vocabSize; j++)
                if (logits.Data[i][j] > maxLogit)
                    maxLogit = logits.Data[i][j];

            double sumExp = 0;
            for (int j = 0; j < vocabSize; j++)
                sumExp += Math.Exp (logits.Data[i][j] - maxLogit);

            loss -= logits.Data[i][targets[i]] - maxLogit - Math.Log (sumExp);
        }

        return loss / seqLength;
    }

    // Backward pass (computes gradients)
    public static Matrix Backward (Matrix logits, int[] targets) {
        int seqLength = logits.Rows;
        int vocabSize = logits.Cols;
        Matrix grad = new Matrix (seqLength, vocabSize);

        for (int i = 0; i < seqLength; i++) {
            double maxLogit = double.MinValue;
            for (int j = 0; j < vocabSize; j++)
                if (logits.Data[i][j] > maxLogit)
                    maxLogit = logits.Data[i][j];

            double sumExp = 0;
            double[] expValues = new double[vocabSize];
            for (int j = 0; j < vocabSize; j++) {
                expValues[j] = Math.Exp (logits.Data[i][j] - maxLogit);
                sumExp += expValues[j];
            }

            for (int j = 0; j < vocabSize; j++) {
                double softmax = expValues[j] / sumExp;
                grad.Data[i][j] = softmax;
            }

            grad.Data[i][targets[i]] -= 1.0;
        }

        // Average gradients over sequence length
        for (int i = 0; i < seqLength; i++)
        for (int j = 0; j < vocabSize; j++)
            grad.Data[i][j] /= seqLength;

        return grad;
    }
}
