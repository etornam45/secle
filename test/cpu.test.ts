
import { assertEquals } from "@std/assert";
import { CPU } from "../core/backends/cpu.ts";
import { Tensor } from "../core/tensor.ts";

Deno.test("CPU Backend Availability", () => {
  if (!CPU.is_available()) {
    throw new Error("CPU backend should be available.");
  }
});

Deno.test("CPU Matrix Multiplication", () => {
  const a = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3], false, "cpu");
  const b = Tensor.fromArray([7, 8, 9, 10, 11, 12], [3, 2], false, "cpu");
  const result = CPU.matmul(a, b);
  const expected = Tensor.fromArray([58, 64, 139, 154], [2, 2]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor Addition", () => {
  const a = Tensor.fromArray([1, 2, 3], [3], false, "cpu");
  const b = Tensor.fromArray([4, 5, 6], [3], false, "cpu");
  const result = CPU.add(a, b);
  const expected = Tensor.fromArray([5, 7, 9], [3]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor Subtraction", () => {
  const a = Tensor.fromArray([5, 7, 9], [1, 3], false, "cpu");
  const b = Tensor.fromArray([1, 2, 3], [1, 3], false, "cpu");
  const result = CPU.sub(a, b);
  const expected = Tensor.fromArray([4, 5, 6], [1, 3]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor-Scalar Addition", () => {
  const a = Tensor.fromArray([1, 2, 3], [3], false, "cpu");
  const scalar = 5;
  const result = CPU.add_a_number(a, scalar);
  const expected = Tensor.fromArray([6, 7, 8], [3]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor-Scalar Multiplication", () => {
  const a = Tensor.fromArray([1, 2, 3], [3], false, "cpu");
  const scalar = 4;
  const result = CPU.mul_scalar(a, scalar);
  const expected = Tensor.fromArray([4, 8, 12], [3]);
  assertEquals(result.data, expected.data);
});


Deno.test("CPU Tensor Power", () => {
  const a = Tensor.fromArray([1, 2, 4], [3], false, "cpu");
  const exponent = 3;
  const result = CPU.pow(a, exponent);
  const expected = Tensor.fromArray([1, 8, 64], [3]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor Negation", () => {
  const a = Tensor.fromArray([1, -2, 3], [3], false, "cpu");
  const result = CPU.neg(a);
  const expected = Tensor.fromArray([-1, 2, -3], [3]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor Mean", () => {
  const a = Tensor.fromArray([1, 2, 3, 4], [2, 2], false, "cpu");
  const result = CPU.mean(a);
  const expected = Tensor.fromArray([2.5], [1]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor Sum", () => {
  const a = Tensor.fromArray([1, 2, 3, 4], [2, 2], false, "cpu");
  const result = CPU.sum(a);
  const expected = Tensor.fromArray([10], [1]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU ReLU Activation", () => {
  const a = Tensor.fromArray([-1, 2, -3, 4], [4], false, "cpu");
  const result = CPU.relu(a);
  const expected = Tensor.fromArray([0, 2, 0, 4], [4]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor Negation", () => {
  const a = Tensor.fromArray([1, -2, 3], [3], false, "cpu");
  const result = CPU.neg(a);
  const expected = Tensor.fromArray([-1, 2, -3], [3]);
  assertEquals(result.data, expected.data);
});


Deno.test("CPU Tensor Division", () => {
  const a = Tensor.fromArray([10, 20, 30], [3], false, "cpu");
  const b = Tensor.fromArray([2, 4, 5], [3], false, "cpu");
  const result = CPU.div(a, b);
  const expected = Tensor.fromArray([5, 5, 6], [3]);
  assertEquals(result.data, expected.data);
});

Deno.test("CPU Tensor Summation", () => {
  const a = Tensor.fromArray([1, 2, 3, 4], [2, 2], false, "cpu");
  const result = CPU.sum(a);
  const expected = Tensor.fromArray([10], [1]);
  assertEquals(result.data, expected.data);
});