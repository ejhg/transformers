using mingpt3;
using transformers.entrypoints;

var command = args[0];

Console.WriteLine ($"Running {command}");

if (command.StartsWith ("llama.cs:")) {
    llama_cs.run (command);
} else if (command.StartsWith ("llama.cs.svd:")) {
    llama_cs_svd.run (command);
} else if (command.StartsWith ("llama.torchsharp:")) {
    llama_touchsharp.run (command);
} else if (command.StartsWith ("mingpt:train")) {
    mingpt_train.run (command);
}

Console.WriteLine ("done");
