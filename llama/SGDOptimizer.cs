namespace llama;

public class SGDOptimizer
{
    public double LearningRate;

    public SGDOptimizer (double learningRate) {
        LearningRate = learningRate;
    }

    public void Step (LlamaForCausalLM model) {
        // Update TokenEmbedding weights
        for (int i = 0; i < model.TokenEmbedding.Weights.GetLength (0); i++)
        for (int j = 0; j < model.TokenEmbedding.Weights.GetLength (1); j++) {
            model.TokenEmbedding.Weights[i, j] -= LearningRate * model.TokenEmbedding.Gradients[i, j];
            model.TokenEmbedding.Gradients[i, j] = 0; // Reset gradients
        }

        // Update OutputProjection
        for (int i = 0; i < model.VocabSize; i++)
        for (int j = 0; j < model.EmbedSize; j++) {
            model.OutputProjection[i, j] -= LearningRate * model.dOutputProjection[i, j];
            model.dOutputProjection[i, j] = 0;
        }

        // Update LayerNorm parameters
        UpdateLayerNorm (model.FinalLayerNorm);

        // Update TransformerBlocks
        foreach (var block in model.TransformerBlocks) {
            UpdateLayerNorm (block.Norm1);
            UpdateLayerNorm (block.Norm2);

            // Update SelfAttention parameters
            UpdateMatrix (block.SelfAttention.Wq, block.SelfAttention.dWq);
            UpdateMatrix (block.SelfAttention.Wk, block.SelfAttention.dWk);
            UpdateMatrix (block.SelfAttention.Wv, block.SelfAttention.dWv);
            UpdateMatrix (block.SelfAttention.Wo, block.SelfAttention.dWo);

            // Reset gradients
            ZeroMatrix (block.SelfAttention.dWq);
            ZeroMatrix (block.SelfAttention.dWk);
            ZeroMatrix (block.SelfAttention.dWv);
            ZeroMatrix (block.SelfAttention.dWo);

            // Update FeedForward parameters
            UpdateFeedForward (block.FeedForward);
        }
    }

    private void UpdateLayerNorm (LayerNorm layerNorm) {
        for (int i = 0; i < layerNorm.Size; i++) {
            layerNorm.Gamma[i] -= LearningRate * layerNorm.dGamma[i];
            layerNorm.Beta[i] -= LearningRate * layerNorm.dBeta[i];
            layerNorm.dGamma[i] = 0;
            layerNorm.dBeta[i] = 0;
        }
    }

    private void UpdateFeedForward (FeedForward feedForward) {
        // Update W1 and B1
        for (int i = 0; i < feedForward.HiddenSize; i++) {
            feedForward.B1[i] -= LearningRate * feedForward.dB1[i];
            for (int j = 0; j < feedForward.EmbedSize; j++) {
                feedForward.W1[i, j] -= LearningRate * feedForward.dW1[i, j];
                feedForward.dW1[i, j] = 0;
            }

            feedForward.dB1[i] = 0;
        }

        // Update W2 and B2
        for (int i = 0; i < feedForward.EmbedSize; i++) {
            feedForward.B2[i] -= LearningRate * feedForward.dB2[i];
            for (int j = 0; j < feedForward.HiddenSize; j++) {
                feedForward.W2[i, j] -= LearningRate * feedForward.dW2[i, j];
                feedForward.dW2[i, j] = 0;
            }

            feedForward.dB2[i] = 0;
        }
    }

    private void UpdateMatrix (double[,] weights, double[,] gradients) {
        int rows = weights.GetLength (0);
        int cols = weights.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            weights[i, j] -= LearningRate * gradients[i, j];
            gradients[i, j] = 0;
        }
    }

    private void ZeroMatrix (double[,] matrix) {
        int rows = matrix.GetLength (0);
        int cols = matrix.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = 0;
    }
}
