namespace mingpt.cs;

public class Optimizer
{
    double LearningRate;

    public Optimizer (double learningRate) {
        LearningRate = learningRate;
    }

    public void Step (Model model) {
        model.TokenEmbedding.UpdateParameters (LearningRate);
        model.PositionalEmbedding.UpdateParameters (LearningRate);

        model.FinalLayer.UpdateParameters (LearningRate);

        // Update transformer layers
        foreach (var layer in model.Layers) {
            // Update self-attention parameters
            layer.SelfAttention.Wq -= LearningRate * layer.SelfAttention.GradWq;
            layer.SelfAttention.Wk -= LearningRate * layer.SelfAttention.GradWk;
            layer.SelfAttention.Wv -= LearningRate * layer.SelfAttention.GradWv;
            layer.SelfAttention.Wo -= LearningRate * layer.SelfAttention.GradWo;

            layer.SelfAttention.GradWq.Clear ();
            layer.SelfAttention.GradWk.Clear ();
            layer.SelfAttention.GradWv.Clear ();
            layer.SelfAttention.GradWo.Clear ();

            layer.FFN.Linear1.UpdateParameters (LearningRate);
            layer.FFN.Linear2.UpdateParameters (LearningRate);

            layer.LayerNorm1.UpdateParameters (LearningRate);
            layer.LayerNorm2.UpdateParameters (LearningRate);
        }
    }
}
