using mingpt.torchsharp;

namespace transformers.entrypoints;

static class mingpt_torchsharp
{
    public static void run (string command) {
        train ("resources/tinyshakespeare.txt");
    }

    public static void train (string sourceFile) {
        Console.WriteLine ("MinGPT TorchSharp Training");
        Console.WriteLine ("==========================");

        try {
            Example.Train (sourceFile);
        } catch (Exception ex) {
            Console.WriteLine ($"Error during training: {ex.Message}");
            Console.WriteLine (ex.StackTrace);
        }
    }
}
