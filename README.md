
# micrograd

This repo adds an interactive browser-based visualizer on top of Andrej Karpathy's micrograd engine. For everything about how the neural network and autograd engine are implemented, see [karpathy/micrograd](https://github.com/karpathy/micrograd). What lives in `viz/index.html` is a single self-contained HTML file (no build step, no dependencies) that ports the full `Value`, `Neuron`, `Layer`, and `MLP` classes to JavaScript and teaches you how they work through four interactive sections:

- **Computation graph**: builds `L = (a + b) x c` as a directed acyclic graph rendered in SVG, with a forward pass that lights up node values in blue and a backward pass that propagates gradients in gold, hover any node to inspect its exact `data` and `grad`
- **Single neuron**: SVG circuit of `ReLU(w1*x1 + w2*x2 + b)` where you can activate the neuron and then backprop through it, with weight gradients shown on the edges and a small ReLU curve drawn inline
- **MLP architecture**: SVG of a `2 -> 4 -> 4 -> 1` network where edge thickness encodes weight magnitude, edge color encodes sign (teal for positive, red for negative), and gold rings appear on neurons with high average gradient after the backward pass
- **Training loop**: watches a `2 -> 8 -> 8 -> 1` network learn to separate two concentric rings using SVM hinge loss and SGD, with a live canvas loss curve, a canvas decision boundary that updates every step, and running accuracy and loss stats

To open it, run `python3 -m http.server 8080` from the `viz/` directory and go to `http://localhost:8080`.

### License

MIT
