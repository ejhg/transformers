namespace mingpt3;

static class CrossEntropyLoss
{
    public static double ComputeLoss (Matrix logits, int[] targetIds, out Matrix dLogitsBatch) {
        int N = targetIds.Length;
        int V = logits.Cols;
        var dLogits = new Matrix (N, V);
        double loss = 0.0;

        for (int i = 0; i < N; i++) {
            double maxLogit = double.NegativeInfinity;
            for (int j = 0; j < V; j++)
                if (logits.Data[i, j] > maxLogit)
                    maxLogit = logits.Data[i, j];

            double sumExp = 0.0;
            for (int j = 0; j < V; j++)
                sumExp += Math.Exp (logits.Data[i, j] - maxLogit);
            double logSumExp = maxLogit + Math.Log (sumExp);

            loss -= logits.Data[i, targetIds[i]] - logSumExp;

            // Compute gradient
            for (int j = 0; j < V; j++) {
                double softmax = Math.Exp (logits.Data[i, j] - logSumExp) / sumExp;
                dLogits.Data[i, j] = softmax;
            }

            dLogits.Data[i, targetIds[i]] -= 1.0;
        }

        // Normalize gradient
        for (int i = 0; i < N; i++)
        for (int j = 0; j < V; j++)
            dLogits.Data[i, j] /= N;

        dLogitsBatch = dLogits;

        return loss / N;
    }
}
