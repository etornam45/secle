import { assertEquals } from "@std/assert";
import { WebGPU } from "../core/backends/webgpu.ts";
import { Tensor } from "../core/tensor.ts";



Deno.test("WebGPU Backend Availability", () => {
  if (!WebGPU.is_available()) {
    throw new Error("WebGPU backend should be available.");
  }
});

Deno.test("WebGPU Matrix Multiplication", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3], false, "webgpu");
  const b = Tensor.fromArray([7, 8, 9, 10, 11, 12], [3, 2], false, "webgpu");
  const result = WebGPU.matmul(a, b);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([58, 64, 139, 154], [2, 2]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor Addition", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([1, 2, 3], [3], false, "webgpu");
  const b = Tensor.fromArray([4, 5, 6], [3], false, "webgpu");
  const result = WebGPU.add(a, b);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([5, 7, 9], [3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor Subtraction", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([5, 7, 9], [1, 3], false, "webgpu");
  const b = Tensor.fromArray([1, 2, 3], [1, 3], false, "webgpu");
  const result = WebGPU.sub(a, b);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([4, 5, 6], [1, 3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor-Scalar Addition", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([1, 2, 3], [3], false, "webgpu");
  const scalar = 5;
  const result = WebGPU.add_a_number(a, scalar);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([6, 7, 8], [3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor-Scalar Multiplication", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([1, 2, 3], [3], false, "webgpu");
  const scalar = 4;
  const result = WebGPU.mul_scalar(a, scalar);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([4, 8, 12], [3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor Power", async () => {
  // NOTE: The GPU results may have minor floating-point differences compared to CPU results
  await WebGPU.init();
  const a = Tensor.fromArray([1, 2, 4], [3], false, "webgpu");
  const exponent = 3;
  const result = WebGPU.pow(a, exponent);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([1, 8, 64], [3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor Mean", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([1, 2, 3, 4], [2, 2], false, "webgpu");
  const result = WebGPU.mean(a);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([2.5], [1]);
  assertEquals(resultData, expected.data);
});


Deno.test("WebGPU ReLU Activation", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([-1, 2, -3, 4], [4], false, "webgpu");
  const result = WebGPU.relu(a);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([0, 2, 0, 4], [4]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor Negation", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([1, -2, 3], [3], false, "webgpu");
  const result = WebGPU.neg(a);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([-1, 2, -3], [3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor Division", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([10, 20, 30], [3], false, "webgpu");
  const b = Tensor.fromArray([2, 4, 5], [3], false, "webgpu");
  const result = WebGPU.div(a, b);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([5, 5, 6], [3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor Summation", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([1, 2, 3, 4], [2, 2], false, "webgpu");
  const result = WebGPU.sum(a);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([10], [1]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU tanh Activation", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([-1, 0, 1], [3], false, "webgpu");
  const result = WebGPU.tanh(a);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([Math.tanh(-1), Math.tanh(0), Math.tanh(1)], [3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Sigmoid Activation", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([-1, 0, 1], [3], false, "webgpu");
  const result = WebGPU.sigmoid(a);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([1 / (1 + Math.exp(1)), 0.5, 1 / (1 + Math.exp(-1))], [3]);
  assertEquals(resultData, expected.data);
});

Deno.test("WebGPU Tensor-Scalar Multiplication (mul_scalar)", async () => {
  await WebGPU.init();
  const a = Tensor.fromArray([1, 2, 3], [3], false, "webgpu");
  const scalar = 3;
  const result = WebGPU.mul_scalar(a, scalar);
  const resultData = await WebGPU.readTensorData(result);
  const expected = Tensor.fromArray([3, 6, 9], [3]);
  assertEquals(resultData, expected.data);
});

