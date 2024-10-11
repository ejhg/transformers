namespace mingpt7;

public class TransformerBlock
{
    public int embeddingDim;
    public MultiHeadSelfAttention selfAttention;
    public LayerNorm layerNorm1;
    public FeedForward feedForward;
    public LayerNorm layerNorm2;

    public TransformerBlock (int embeddingDim, int numHeads, int hiddenDim) {
        this.embeddingDim = embeddingDim;
        selfAttention = new MultiHeadSelfAttention (numHeads, embeddingDim);
        layerNorm1 = new LayerNorm (embeddingDim);
        feedForward = new FeedForward (embeddingDim, hiddenDim);
        layerNorm2 = new LayerNorm (embeddingDim);
    }

    public double[][] Forward (double[][] x) {
        double[][] attnOut = selfAttention.Forward (x);
        double[][] x1 = new double[x.Length][];
        for (int t = 0; t < x.Length; t++) {
            x1[t] = math.Add (x[t], attnOut[t]);
            x1[t] = layerNorm1.Forward (x1[t]);
        }

        double[][] ffOut = new double[x.Length][];
        for (int t = 0; t < x.Length; t++) {
            ffOut[t] = feedForward.Forward (x1[t]);
        }

        double[][] outp = new double[x.Length][];
        for (int t = 0; t < x.Length; t++) {
            outp[t] = math.Add (x1[t], ffOut[t]);
            outp[t] = layerNorm2.Forward (outp[t]);
        }

        return outp;
    }

    public double[][] Backward (double[][] gradOutput) {
        double[][] gradFF = new double[gradOutput.Length][];
        for (int t = 0; t < gradOutput.Length; t++) {
            double[] gradLN = layerNorm2.Backward (gradOutput[t]);
            gradFF[t] = feedForward.Backward (gradLN);
        }

        double[][] gradAttnInput = new double[gradOutput.Length][];
        for (int t = 0; t < gradOutput.Length; t++) {
            double[] gradResidual = math.Add (gradFF[t], gradOutput[t]);
            double[] gradLN = layerNorm1.Backward (gradResidual);
            gradAttnInput[t] = gradLN;
        }

        double[][] gradSelfAttn = selfAttention.Backward (gradAttnInput);
        return gradSelfAttn;
    }

    public void UpdateParameters (double learningRate) {
        selfAttention.UpdateParameters (learningRate);
        layerNorm1.UpdateParameters (learningRate);
        feedForward.UpdateParameters (learningRate);
        layerNorm2.UpdateParameters (learningRate);
    }
}
