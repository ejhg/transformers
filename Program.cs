/**
 * Directories:
 * - Current working directory must be repo root so that resources folder may be found.
 * - Assume that llama models are located in ~/home/llama
 */

Console.WriteLine ("start");

var HOME = Environment.GetFolderPath (Environment.SpecialFolder.UserProfile);

// llama.cs.Entrypoint.main ("resources/stories15M.bin", "-s", "1");

llama.torchsharp.Entrypoint.run (
    // Assume that llama models are located in ~/home/llama
    HOME + "/llama/llama-2-7b-chat",
    "llama.torchsharp/resources");

Console.WriteLine ("done");
