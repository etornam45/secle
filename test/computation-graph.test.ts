import { assertEquals } from "@std/assert";
import * as nn from "../index.ts";

class Linear extends nn.Module {
  weights: nn.Tensor;
  bias: nn.Tensor;

  constructor(inputSize: number = 10, outputSize: number = 10) {
    super();
    this.weights = nn.Tensor.ones([inputSize, outputSize], true);
    this.bias = nn.Tensor.zeros([outputSize], true);
    this.register_parameters();
  }

  override forward(x: nn.Tensor): nn.Tensor {
    x = nn.matmul(x, this.weights);
    x = nn.add(x, this.bias);
    x = nn.relu(x);
    return x;
  }
}



Deno.test("Test Linear Layer", () => {
  const layer = new Linear();
  const input = nn.Tensor.ones([1, 10]);
  const output = layer.forward(input);
  assertEquals(output.shape, [1, 10]);
});

Deno.test("Test Gradient Shape", () => {
  const input = nn.Tensor.ones([1, 2]);
  const layer = new Linear(2, 1);
  const output = layer.forward(input);
  output.backward();
  const grad = layer.weights._grad!;
  assertEquals(grad.shape, [2, 1]);
});

Deno.test("Test Weight Gradient Values", () => {
  const input = nn.Tensor.ones([1, 2]);
  const layer = new Linear(2, 1);
  const output = layer.forward(input);
  output.backward();
  const grad = layer.weights._grad!;
  assertEquals(Array.from(grad.data), [1, 1]);
});

Deno.test("Test Bias Gradient Values", () => {
  const input = nn.Tensor.ones([1, 2]);
  const layer = new Linear(2, 1);
  const output = layer.forward(input);
  output.backward();
  const grad = layer.bias._grad!;
  assertEquals(Array.from(grad.data), [1]);
});

Deno.test("Test Multiple Forward Passes", () => {
  const input = nn.Tensor.ones([1, 2]);
  const layer = new Linear(2, 1);
  const output1 = layer.forward(input);
  const output2 = layer.forward(input);
  assertEquals(output1.shape, output2.shape);
  assertEquals(Array.from(output1.data), Array.from(output2.data));
});


// Test nn.Matmul a @ b = c where a: [m, n], b: [n, p], c: [m, p]
Deno.test("Test Linear Layer Matmul", () => {
  const a = nn.Tensor.ones([1, 2]);
  const b = nn.Tensor.ones([2, 1]);
  const c = nn.matmul(a, b);
  assertEquals(c.shape, [1, 1]);
  assertEquals(Array.from(c.data), [2]);
});

// Test nn.Matmul _backward function
Deno.test("Test Linear Layer Matmul Backward", () => {
  const a = nn.Tensor.ones([1, 2], true);
  const b = nn.Tensor.ones([2, 1], true);
  const c = nn.matmul(a, b);
  c.backward();
  const a_grad = a._grad!;
  const b_grad = b._grad!;
  assertEquals(a_grad.shape, [1, 2]);
  assertEquals(b_grad.shape, [2, 1]);
  assertEquals(Array.from(a_grad.data), [1, 1]);
  assertEquals(Array.from(b_grad.data), [1, 1]);
});