import * as nn from "../index.ts";
import { SGD } from "../core/optim.ts";

class Model extends nn.Module {
  linear: nn.Linear;
  linear2: nn.Linear;

  constructor(input_size: number, output_size: number) {
    super();
    this.linear = new nn.Linear(input_size, 10);
    this.linear2 = new nn.Linear(10, output_size);
    this.register_parameters([this.linear, this.linear2]);
  }

  override forward(x: nn.Tensor): nn.Tensor {
    return this.linear2.$(this.linear.$(x));
  }
}

// Simple linear data y = 2x + 1. Range of x is [-10, 10].
// create the data and put it in a tensor

function create_data(start: number, end: number, num_points: number): { x: nn.Tensor, y: nn.Tensor }[] {
  const data = [];
  for (let i = 0; i < num_points; i++) {
    const x = nn.Tensor.fromArray([start + (end - start) * i / (num_points - 1)], [1, 1], false); // Input data shouldn't require gradients
    const y = nn.add(nn.mul(x, 2), 1);
    data.push({ x: x, y: y });
  }
  return data;
}


const data = create_data(-10, 10, 100);

console.log(data[0].x.shape);
console.log(data[0].y.shape);

const model = new Model(1, 1);
const optimizer = new SGD(model.parameters, 0.001);

let old_parameters = snapshot_params(model.parameters);

for (let i = 0; i < 100; i++) {
  let epoch_loss: number[] = [];
  for (const { x, y } of data) {
    const output = model.$(x);
    const loss = nn.mean(nn.pow(nn.sub(output, y), 2));
    loss.backward();
    optimizer.step();
    epoch_loss.push(loss.data[0]);
    optimizer.zero_grad();
  }
  const param_diff = param_diff_from_snapshot(old_parameters, model.parameters);
  old_parameters = snapshot_params(model.parameters);
  console.log("Param diff:", param_diff);
  console.log("Epoch loss:", epoch_loss.reduce((a, b) => a + b, 0) / epoch_loss.length);
  epoch_loss = [];
}

function param_diff_from_snapshot(before: Float32Array[], params: nn.Tensor[]): number {
  let diff = 0;
  for (let i = 0; i < params.length; i++) {
    const a = before[i];
    const b = params[i].data;
    let d = 0;
    for (let j = 0; j < a.length; j++) d += (a[j] - b[j]) ** 2;
    diff += d;
  }
  return diff;
}

function snapshot_params(params: nn.Tensor[]): Float32Array[] {
  return params.map(p => Float32Array.from(p.data));
}