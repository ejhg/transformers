using static TorchSharp.torch;

namespace mingpt.torchsharp;

public static class Example
{
    public static void Train (string text_file_path = "resources/tinyshakespeare.txt") {
        // Read text file
        var text = File.ReadAllText (text_file_path);

        Console.WriteLine ($"Text length: {text.Length} characters");

        // Create dataset
        var block_size = 128;
        var dataset = new CharDataset (text, block_size);
        Console.WriteLine ($"Dataset length: {dataset.Count}");
        Console.WriteLine ($"Vocab size: {dataset.VocabSize}");

        // Create model configuration
        var config = Utils.GetDefaultConfig ();
        config.vocab_size = dataset.VocabSize;
        config.block_size = block_size;
        config.n_layer = 6; // smaller model for testing
        config.n_head = 6;
        config.n_embd = 192;

        // Create model
        var model = new GPT (config);
        Console.WriteLine ($"Model created with {model.parameters ().Sum (p => p.numel ())} parameters");

        // Create trainer configuration
        var trainer_config = Utils.GetDefaultTrainerConfig ();
        trainer_config.max_iters = 10000;
        trainer_config.batch_size = 4;
        trainer_config.learning_rate = 1e-3;

        // Create trainer
        var trainer = new Trainer (trainer_config, model, dataset);

        // Add callback for generation during training
        trainer.AddCallback ("on_batch_end", t => {
            if (t.iter_num % 20 == 0) {
                // Generate some text
                model.eval ();
                using (no_grad ()) {
                    var context = "hello";
                    var x = tensor (dataset.Encode (context), dtype: ScalarType.Int64).unsqueeze (0);
                    var y = model.Generate (x, max_new_tokens: 50, temperature: 1.0, do_sample: true, top_k: 10);
                    var completion = dataset.Decode (y.squeeze (0).data<long> ().Select (x => (int)x).ToArray ());
                    Console.WriteLine ($"Generated: {completion}");
                }

                model.train ();
            }
        });

        // Train the model
        Console.WriteLine ("Starting training...");
        trainer.Run ();

        Console.WriteLine ("Training completed!");

        // Final generation
        Console.WriteLine ("\nFinal generation:");
        model.eval ();
        using (no_grad ()) {
            var context = "The ";
            var x = tensor (dataset.Encode (context), dtype: ScalarType.Int64).unsqueeze (0);
            var y = model.Generate (x, max_new_tokens: 100, temperature: 1.0, do_sample: true, top_k: 10);
            var completion = dataset.Decode (y.squeeze (0).data<long> ().Select (x => (int)x).ToArray ());
            Console.WriteLine (completion);
        }
    }
}
