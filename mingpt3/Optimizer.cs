namespace mingpt3;

public class Optimizer
{
    public double LearningRate;

    public Optimizer (double learningRate) {
        LearningRate = learningRate;
    }

    public void Step (GPTModel model) {
        // Update token embeddings
        model.TokenEmbedding.Weights -= LearningRate * model.TokenEmbedding.GradWeights;
        model.TokenEmbedding.GradWeights.Clear ();

        // Update positional embeddings
        model.PositionalEmbedding.Weights -= LearningRate * model.PositionalEmbedding.GradWeights;
        model.PositionalEmbedding.GradWeights.Clear ();

        // Update final linear layer
        model.FinalLayer.Weights -= LearningRate * model.FinalLayer.GradWeights;
        model.FinalLayer.Bias -= LearningRate * model.FinalLayer.GradBias;
        model.FinalLayer.GradWeights.Clear ();
        model.FinalLayer.GradBias.Clear ();

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

            // Update feed-forward network parameters
            layer.FFN.Linear1.Weights -= LearningRate * layer.FFN.Linear1.GradWeights;
            layer.FFN.Linear1.Bias -= LearningRate * layer.FFN.Linear1.GradBias;
            layer.FFN.Linear2.Weights -= LearningRate * layer.FFN.Linear2.GradWeights;
            layer.FFN.Linear2.Bias -= LearningRate * layer.FFN.Linear2.GradBias;

            layer.FFN.Linear1.GradWeights.Clear ();
            layer.FFN.Linear1.GradBias.Clear ();
            layer.FFN.Linear2.GradWeights.Clear ();
            layer.FFN.Linear2.GradBias.Clear ();

            // Update layer norm parameters
            for (int i = 0; i < layer.LayerNorm1.EmbeddingSize; i++) {
                layer.LayerNorm1.Gamma[i] -= LearningRate * layer.LayerNorm1.GradGamma[i];
                layer.LayerNorm1.Beta[i] -= LearningRate * layer.LayerNorm1.GradBeta[i];
                layer.LayerNorm1.GradGamma[i] = 0.0;
                layer.LayerNorm1.GradBeta[i] = 0.0;

                layer.LayerNorm2.Gamma[i] -= LearningRate * layer.LayerNorm2.GradGamma[i];
                layer.LayerNorm2.Beta[i] -= LearningRate * layer.LayerNorm2.GradBeta[i];
                layer.LayerNorm2.GradGamma[i] = 0.0;
                layer.LayerNorm2.GradBeta[i] = 0.0;
            }
        }
    }
}
