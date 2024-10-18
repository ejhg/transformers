using llama;

using transformers.utils;

public class LlamaTest
{
    public static void run () {
        int embeddingSize = 128;
        var hiddenSize = 4 * embeddingSize;
        int numHeads = 6;
        int maxSeqLen = 16;
        int batchSize = 64;

        var data = DataLoader.LoadData (maxSeqLen, out var vocabSize, out var vocabulary);

        var model = new LlamaForCausalLM (
            vocabSize,
            embeddingSize,
            hiddenSize,
            hiddenSize / numHeads,
            numLayers: 4);
        var optimizer = new SGDOptimizer (learningRate: 0.0005);

        Trainer.train (
            model,
            optimizer,
            data,
            1000,
            batchSize,
            callback: () => {
                var a = TextGeneration.predict (_ => model.Forward (_), vocabulary, "The ", maxSeqLen, topK: 10);
                var b = TextGeneration.predict (_ => model.Forward (_), vocabulary, "The ", maxSeqLen, topK: 0);
                var c = TextGeneration.predict (_ => model.Forward (_), vocabulary, "The ", maxSeqLen, topK: 0, argmax: true);

                Console.WriteLine (a);
                Console.WriteLine (b);
                Console.WriteLine (c);
            });
    }
}
