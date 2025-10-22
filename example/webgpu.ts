import { WebGPU } from "../core/backends/webgpu.ts";
import { CPU } from "../core/backends/cpu.ts";
import * as nn from "../index.ts";

const size = 1000;

// Demo: manual WebGPU add (not integrated into ops)
async function demo_webgpu_add() {
  try {
    await WebGPU.init();
  } catch (_e) {
    console.log("WebGPU not available, skipping demo.");
    return;
  }
  const a = nn.Tensor.randn([size, size], false, "webgpu");
  a._name = "A";
  const b = nn.Tensor.randn([size, size], false, "webgpu");
  b._name = "B";
  const out = WebGPU.matmul(WebGPU.add(WebGPU.matmul(WebGPU.matmul(WebGPU.add(a, b), b), a), b), a);
  const data = await WebGPU.readTensorData(out);
  // console.log("WebGPU add result:", Array.from(data));
}

function demo_cpu_add() {
  const a = nn.Tensor.randn([size, size], false, "cpu");
  a._name = "A"
  const b = nn.Tensor.randn([size, size], false, "cpu");
  b._name = "B"
  const out = CPU.matmul(CPU.add(CPU.matmul(CPU.matmul(CPU.add(a, b), b), a), b), a);
  // console.log("CPU add result:", Array.from(out.data));
}

let st = Date.now();
demo_cpu_add();
console.log("CPU add time (ms):", Date.now() - st);

st = Date.now();
await demo_webgpu_add();
console.log("WebGPU add time (ms):", Date.now() - st);