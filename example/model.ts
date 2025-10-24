import * as nn from "../index.ts";
import { SGD } from "../core/optim.ts";

class Model extends nn.Module {
  seq: nn.Sequencial;
  
  constructor(input_size: number, output_size: number, hidden_dim = 10) {
    super();
    this.seq = new nn.Sequencial([
      new nn.Linear(input_size, hidden_dim),
      new nn.Sigmoid(),
      // new nn.Linear(hidden_dim, hidden_dim),
      // new nn.Sigmoid(),
      // new nn.Linear(hidden_dim, hidden_dim),
      // new nn.Sigmoid(),
      // new nn.Linear(hidden_dim, hidden_dim),
      // new nn.Sigmoid(),
      // new nn.Linear(hidden_dim, hidden_dim),
      // new nn.Sigmoid(),
      new nn.Linear(hidden_dim, hidden_dim),
      new nn.Sigmoid(),
      new nn.Linear(hidden_dim, hidden_dim),
      new nn.Sigmoid(),
      new nn.Linear(hidden_dim, output_size)
    ])
    this.register_parameters([this.seq]);
  }

  override forward(x: nn.Tensor): nn.Tensor {
    return this.seq.$(x);
  }
}

// Simple linear data y = sin(x). Range of x is [-10, 10].
// create the data and put it in a tensor

function create_data(start: number, end: number, num_points: number): { x: nn.Tensor, y: nn.Tensor }[] {
  const data = [];
  for (let i = 0; i < num_points; i++) {
    const x = nn.Tensor.fromArray([start + (end - start) * i / (num_points - 1)], [1, 1], false); // Input data shouldn't require gradients
    // sin(x)
    const y = new nn.Tensor([Math.sin(x.data[0])], [1, 1], false);
    data.push({ x: x, y: y });
  }
  return data;
}


const data = create_data(-10, 10, 500);

console.log(data[0].x.shape);
console.log(data[0].y.shape);

const device: nn.Device = "webgpu"
const model = new Model(1, 1, 1000);
await model.to(device)
const optimizer = new SGD(model.parameters, 0.000001);
const criterion = nn.MSELoss()


console.log("Training started...on ", device);
let total_losses: number[] = [];
for (let i = 0; i < 10; i++) {
  const start = performance.now();
  let total_loss = nn.Tensor.zeros([1, 1], false, device);
  for (const { x, y } of data) {
    await x.to(device)
    await y.to(device)
    const output = model.$(x);
    const loss = criterion(output, y);
    total_loss = nn.add(total_loss, loss) as nn.Tensor;
    loss.backward();
    optimizer.step();
    // epoch_loss.push(los_val);
    optimizer.zero_grad();
  }
  let los_val: number
  if (total_loss.device === "webgpu") {
    los_val = (await nn.WebGPU.readTensorData(total_loss))[0]
  } else {
    los_val = total_loss.data[0]
  }
  console.log(`Epoch loss: ${i + 1}/${10}`, los_val / data.length);
  total_losses.push(los_val)
  const end = performance.now();
  console.log(`Epoch Time taken: ${end - start} milliseconds`);
}
console.log("Total loss:", total_losses.reduce((a, b) => a + b, 0));
console.log("Parameters:", model.parameters_count);