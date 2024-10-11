namespace transformers.utils;

public static class ActivationFunctions
{
    public static double[] Softmax (double[] logits) {
        double maxLogit = double.MinValue;
        foreach (var logit in logits)
            if (logit > maxLogit)
                maxLogit = logit;

        double sumExp = 0;
        double[] exps = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++) {
            exps[i] = Math.Exp (logits[i] - maxLogit);
            sumExp += exps[i];
        }

        double[] softmax = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            softmax[i] = exps[i] / sumExp;

        return softmax;
    }
}
