using llama;

using transformers.utils;

public class LlamaTest
{
    public static void run () {
        int embeddingSize = 128;
        var hiddenSize = 4 * embeddingSize;
        int numHeads = 6;
        int maxSeqLen = 32;
        int batchSize = 64;

        var data = DataLoader.LoadData (maxSeqLen, out var vocabSize, out var vocabulary);

        var model = new LlamaForCausalLM (
            vocabSize: vocabSize,
            embedSize: embeddingSize,
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            numLayers: 4,
            numQueryGroups: 2,
            new Random ());
        var optimizer = new AdamOptimizer (learningRate: 0.001);

        Trainer.train (
            model,
            optimizer,
            data,
            1000,
            batchSize,
            callback: () => {
                var a = TextGeneration.predict (_ => model.Forward (_).Last (), vocabulary, "The ", maxSeqLen, topK: 10);
                var b = TextGeneration.predict (_ => model.Forward (_).Last (), vocabulary, "The ", maxSeqLen, topK: 0);

                Console.WriteLine (a);
                Console.WriteLine (b);
            });
    }
}
