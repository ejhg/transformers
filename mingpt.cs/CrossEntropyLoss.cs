namespace mingpt.cs;

static class CrossEntropyLoss
{
    public static Matrix ComputeLoss (Matrix logits, int[] targetIds, out double loss) {
        int seqLength = logits.Rows;
        int vocabSize = logits.Cols;
        var grad = new Matrix (seqLength, vocabSize);

        loss = 0.0;

        for (int i = 0; i < seqLength; i++) {
            double maxLogit = double.MinValue;
            for (int j = 0; j < vocabSize; j++) {
                if (logits.Data[i, j] > maxLogit) {
                    maxLogit = logits.Data[i, j];
                }
            }

            double sumExp = 0.0;
            for (int j = 0; j < vocabSize; j++) {
                sumExp += Math.Exp (logits.Data[i, j] - maxLogit);
            }

            loss -= (logits.Data[i, targetIds[i]] - (maxLogit + Math.Log (sumExp))) / seqLength;

            // Compute gradient
            for (int j = 0; j < vocabSize; j++) {
                grad.Data[i, j] = Math.Exp (logits.Data[i, j] - maxLogit) / sumExp;
            }

            grad.Data[i, targetIds[i]] -= 1.0;
        }

        // Normalize gradient
        for (int i = 0; i < seqLength; i++) {
            for (int j = 0; j < vocabSize; j++) {
                grad.Data[i, j] /= seqLength;
            }
        }

        return grad;
    }
}
