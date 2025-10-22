export type Shape = number[];
export type Device = "webgpu" | "cpu"

export class Tensor {
  data: Float32Array;
  shape: Shape;
  requires_grad: boolean;
  _grad?: Tensor;
  _prev: Tensor[] = [];
  _name?: string;
  _backward?: () => void;

  device: Device = "cpu"

  _data_buffer?: GPUBuffer
  _grad_buffer?: GPUBuffer

  constructor(
    data: Float32Array | number[],
    shape: Shape,
    requires_grad: boolean = false,
    device: Device = "cpu"
  ) {
    if (data.length !== shape.reduce((a, b) => a * b, 1)) {
      throw new Error(`Data length ${data.length} does not match shape ${shape.join(", ")}.`);
    }
    if (data instanceof Float32Array) {
      this.data = data;
    } else {
      this.data = new Float32Array(data);
    }
    this.shape = shape;
    this.requires_grad = requires_grad;
    this.device = device
  }

  static fromArray(arr: number[], shape: Shape, requires_grad: boolean = false, device: Device = "cpu"): Tensor {
    return new Tensor(arr, shape, requires_grad, device);
  }

  static zeros(shape: Shape, requires_grad: boolean = false, device: Device = "cpu"): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape, requires_grad, device);
  }

  static ones(shape: Shape, requires_grad: boolean = false, device: Device = "cpu"): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(1);
    return new Tensor(data, shape, requires_grad, device);
  }

  static randn(shape: Shape, requires_grad: boolean = false, device: Device = "cpu"): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    function randomNormal(size: number): Float32Array {
      const data = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        data[i] = z0;
      }
      return data;
    }
    const data = randomNormal(size);
    return new Tensor(data, shape, requires_grad, device);
  }

  toString(): string {
    
    return `Tensor(shape=[${this.shape}])`;
  }


  backward(grad?: Tensor): void {
    if (!this.requires_grad) {
      throw new Error("Cannot call backward on a tensor that does not require gradients.");
    }
    if (!grad) {
      if (this.size() !== 1) throw new Error('grad must be provided for non-scalar tensors');
      grad = Tensor.ones(this.shape, this.requires_grad);
    }
    this._grad = grad;

    
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const buildTopo = (tensor: Tensor) => {
      if (visited.has(tensor)) return;
      visited.add(tensor);
      for (const child of tensor._prev) {
        buildTopo(child);
      }
      topo.push(tensor);
    }

    buildTopo(this);
    topo.reverse();

    for (let i = 0; i < topo.length; i++) {
      const tensor = topo[i];
      if (tensor._backward) {
        tensor._backward();
      }
    }
  }

  size(): number {
    return this.data.length;
  }

  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("Transpose is only implemented for 2D tensors.");
    }
    const [rows, cols] = this.shape;
    const result = Tensor.zeros([cols, rows], this.requires_grad, this.device);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.data[j * rows + i] = this.data[i * cols + j];
      }
    }
    return result;
  }

  zero_grad() {
    if (this._grad) {
      this._grad.data.fill(0);
      this._grad = undefined;
    }
  }

  item(): number {
    if (this.data.length !== 1) throw new Error(`Tensor is not scalar. Got Shape(${this.shape}) and Lenght(${this.data.length})`);
    return this.data[0];
  }
}
export class Module {
  _name?: string;
  parameters: Tensor[] = [];

  constructor(_name?: string) {
    this._name = _name;
    
    for (const key in this) {
      if (this[key] instanceof Tensor && this[key].requires_grad) {
        this.parameters.push(this[key]);
      } else if (this[key] instanceof Module) {
        
        this.register_parameters(this[key].parameters);
      }
    }
  }

  register_parameters(_parameters?: Tensor[] | Module[]): void {
    if (!_parameters) {
      for (const key in this) {
        if (this[key] instanceof Tensor && this[key].requires_grad) {
          this.parameters.push(this[key]);
        } else if (this[key] instanceof Module) {
          
          this.register_parameters(this[key].parameters);
        }
      }
      return;
    };
    for (const parameter of _parameters) {
      if (parameter instanceof Tensor) {
        if (parameter.requires_grad) {
          this.parameters.push(parameter);
        } else {
          throw new Error("Parameter does not require gradients.");
        }
      } else if (parameter instanceof Module) {
        this.register_parameters(parameter.parameters);
      } else {
        throw new Error("Parameter is not a Tensor or Module.");
      }
    }
  }

  forward(_input: Tensor): Tensor {
    throw new Error("Method not implemented.");
  }

  backward(_grad: Tensor) {
    throw new Error("Method not implemented.");
  }

  
  $(_input: Tensor): Tensor {
    return this.forward(_input);
  }

  summary(_indent: number = 0): string {
    let summary = "";
    for (const key in this) {
      if (this[key] instanceof Module) {
        summary += `${" ".repeat(_indent * 2)}${this[key]._name ?? "Module"}: ${this[key].summary(_indent + 1)}\n`;
      } else if (this[key] instanceof Tensor) {
        summary += `${" ".repeat(_indent * 2)}${this[key]._name ?? "Parameter"}: ${this[key].shape.join("x")}\n`;
      }
    }
    return summary.trim();
  }

  get parameters_count(): number {
    return this.parameters.map(p => p.size()).reduce((a, b) => a + b, 0);
  }
}
export const CPU = {
  name: "cpu",
  is_available: (): boolean => true,
  matmul: (a: Tensor, b: Tensor): Tensor => {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("Both tensors must be 2-dimensional for matmul." + " Got shapes " + a.shape + " and " + b.shape);
    }
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Incompatible tensor shapes for matmul. Got shapes ${a.shape} and ${b.shape}.`);
    }

    const result = Tensor.zeros([a.shape[0], b.shape[1]], a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.shape[0]; i++) {
      for (let j = 0; j < b.shape[1]; j++) {
        for (let k = 0; k < a.shape[1]; k++) {
          result.data[i * b.shape[1] + j] += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
        }
      }
    }
    return result;
  },

  add: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for addition. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] + b.data[i];
    }
    return result;
  },

  add_a_number: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] + b;
    }
    return result;
  },

  sub: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for subtraction. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] - b.data[i];
    }
    return result;
  },

  mul: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for multiplication. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] * b.data[i];
    }
    return result;
  },

  
  mul_scalar: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] * b;
    }
    return result;
  },

  div: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for division. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] / b.data[i];
    }
    return result;
  },

  neg: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = -a.data[i];
    }
    return result;
  },



  sum: (a: Tensor): Tensor => {
    let result = 0;
    for (let i = 0; i < a.size(); i++) {
      result += a.data[i];
    }
    return new Tensor([result], [1, 1], a.requires_grad);
  },

  mean: (a: Tensor): Tensor => {
    const result = CPU.sum(a);
    return new Tensor([result.data[0] / a.size()], [1, 1], a.requires_grad);
  },

  
  pow: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = Math.pow(a.data[i], b);
    }
    return result;
  },

  relu: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = Math.max(0, a.data[i]);
    }
    return result;
  },

  sigmoid: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = 1 / (1 + Math.exp(-a.data[i]));
    }
    return result;
  },
}
export function matmul(a: Tensor, b: Tensor): Tensor {
  const out = CPU.matmul(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad; 

  out._backward = () => {
    if (!out._grad) return;

    if (a.requires_grad) {
      const a_grad = matmul(out._grad, b.transpose());
      a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
    }

    if (b.requires_grad) {
      const b_grad = matmul(a.transpose(), out._grad);
      b._grad = b._grad ? add(b._grad, b_grad) : b_grad;
    }
  };

  return out;
}

export function sub(a: Tensor, b: Tensor): Tensor {
  const out = CPU.sub(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad;

  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? add(a._grad, out._grad) : out._grad;
    }
    if (b.requires_grad) {
      const negGrad = neg(out._grad);
      b._grad = b._grad ? add(b._grad, negGrad) : negGrad;
    }
  };

  return out;
}

export function neg(a: Tensor): Tensor {
  const out = CPU.neg(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? add(a._grad, neg(out._grad)) : neg(out._grad);
    }
  };
  return out;
}


export function mul(a: Tensor | number, b: Tensor | number): Tensor {

  if (typeof a === "number" && b instanceof Tensor) {
    const out = CPU.mul_scalar(b, a);
    out._prev = [b];
    out.requires_grad = b.requires_grad;
    out._backward = () => {
      if (!out._grad) return;
      if (b.requires_grad) {
        const b_grad = mul(out._grad, a);
        b._grad = b._grad ? add(b._grad, b_grad) : b_grad;
      }
    };
    return out;
  }
  if (typeof b === "number" && a instanceof Tensor) {
    const out = CPU.mul_scalar(a, b);
    out._prev = [a];
    out.requires_grad = a.requires_grad;
    out._backward = () => {
      if (!out._grad) return;
      if (a.requires_grad) {
        const a_grad = mul(out._grad, b);
        a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
      }
    };
    return out;
  }

  if (a instanceof Tensor && b instanceof Tensor) {
    const out = CPU.mul(a, b);
    out._prev = [a, b];
    out.requires_grad = a.requires_grad || b.requires_grad;

    out._backward = () => {
      if (!out._grad) return;
      if (a.requires_grad) {
        const a_grad = mul(out._grad, b);
        a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
      }
      if (b.requires_grad) {
        const b_grad = mul(out._grad, a);
        b._grad = b._grad ? add(b._grad, b_grad) : b_grad;
      }
    };

    return out;
  }
  
  if (typeof a === "number" && typeof b === "number") {
    return new Tensor([a * b], [1, 1]);
  }

  throw new Error("Invalid arguments to mul. Expected a Tensor or a number, got " + typeof a + " and " + typeof b);
}

export function add(a: Tensor | number, b: Tensor | number): Tensor {
  if (typeof a === "number" && b instanceof Tensor) {
    const out = CPU.add_a_number(b, a);
    out._prev = [b];
    out.requires_grad = b.requires_grad;
    out._backward = () => {
      if (!out._grad) return;
      if (b.requires_grad) {
        b._grad = b._grad ? add(b._grad, out._grad) : out._grad;
      }
    };
    return out;
  }
  if (typeof b === "number" && a instanceof Tensor) {
    const out = CPU.add_a_number(a, b);
    out._prev = [a];
    out.requires_grad = a.requires_grad;
    out._backward = () => {
      if (!out._grad) return;
      if (a.requires_grad) {
        a._grad = a._grad ? add(a._grad, out._grad) : out._grad;
      }
    };
    return out;
  }

  if (a instanceof Tensor && b instanceof Tensor) {
    const out = CPU.add(a, b);
    out._prev = [a, b];
    out.requires_grad = a.requires_grad || b.requires_grad;

    out._backward = () => {
      if (!out._grad) return;
      if (a.requires_grad) {
        a._grad = a._grad ? add(a._grad, out._grad) : out._grad;
      }
      if (b.requires_grad) {
        b._grad = b._grad ? add(b._grad, out._grad) : out._grad;
      }
    };

    return out;
  }

  if (typeof a === "number" && typeof b === "number") {
    return new Tensor([a + b], [1, 1]);
  }

  throw new Error("Invalid arguments to add. Expected a Tensor or a number, got " + typeof a + " and " + typeof b);
}


export function mean(a: Tensor): Tensor {
  const out = CPU.mean(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const scale = 1 / a.size();
      const scaled = mul(out._grad, scale);
      const gradBroadcast = new Tensor(
        new Float32Array(a.size()).fill(scaled.data[0]),
        a.shape,
        a.requires_grad,
      );
      a._grad = a._grad ? add(a._grad, gradBroadcast) : gradBroadcast;
    }
  };
  return out;
}

export function pow(a: Tensor, b: number): Tensor {
  const out = CPU.pow(a, b);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const local = mul(CPU.pow(a, b - 1), b);
      const grad = mul(out._grad, local);
      a._grad = a._grad ? add(a._grad, grad) : grad;
    }
  };
  return out;
}

export function sum(a: Tensor): Tensor {
  const out = CPU.sum(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const gradBroadcast = new Tensor(
        new Float32Array(a.size()).fill(out._grad.data[0]),
        a.shape,
        a.requires_grad,
      );
      a._grad = a._grad ? add(a._grad, gradBroadcast) : gradBroadcast;
    }
  };
  return out;
}


export function relu(a: Tensor): Tensor {
  const out = CPU.relu(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const mask = new Tensor(
        new Float32Array(a.data.map(x => x > 0 ? 1 : 0)),
        a.shape,
        a.requires_grad
      );
      const a_grad = mul(out._grad, mask);
      a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
    }
  };
  return out;
}

export function sigmoid(a: Tensor): Tensor {
  const out = CPU.sigmoid(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      
      const ones = Tensor.ones(out.shape, out.requires_grad);
      const sigmoid_derivative = mul(out, sub(ones, out));
      const a_grad = mul(out._grad, sigmoid_derivative);
      a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
    }
  };
  return out;
}
export class Sigmoid extends Module {
  constructor() {
    super()
    
  }

  override forward(_input: Tensor): Tensor {
    return sigmoid(_input)
  }
}

export class Linear extends Module {
  weights: Tensor;
  bias: Tensor;

  constructor(input_size: number, output_size: number) {
    super(`Linear(in_features=${input_size}, out_features=${output_size})`);
    this.weights = Tensor.randn([input_size, output_size], true);
    this.bias = Tensor.randn([1, output_size], true);

    this.register_parameters();
  }

  override forward(x: Tensor): Tensor {
    return add(matmul(x, this.weights), this.bias);
  }
}


export function linear(x: Tensor, weights: Tensor, bias: Tensor): Tensor {
  return add(matmul(x, weights), bias);
}
export interface LossFunction {
  (output: Tensor, y: Tensor): Tensor;
}

export function MSELoss(): LossFunction {
  function criterion(output: Tensor, y: Tensor): Tensor {
    return mean(pow(sub(output, y), 2))
  }
  return criterion
}
export interface Optimizer {
  zero_grad(): void;
  step(): void;
}

export class SGD implements Optimizer {
  private readonly parameters: Tensor[];
  private readonly learningRate: number;
  private readonly momentum: number;
  private readonly maxGradNorm: number;
  
  private readonly velocities: Map<Tensor, Float32Array> = new Map();

  constructor(parameters: Tensor[], learningRate: number, momentum: number = 0, maxGradNorm: number = 1.0) {
    this.parameters = parameters;
    this.learningRate = learningRate;
    this.momentum = momentum;
    this.maxGradNorm = maxGradNorm;
  }

  zero_grad(): void {
    for (const parameter of this.parameters) {
      parameter.zero_grad()
    }
  }

  step(): void {
    
    let totalNorm = 0;
    for (const parameter of this.parameters) {
      if (!parameter._grad) continue;
      const grad = parameter._grad.data;
      for (let i = 0; i < grad.length; i++) {
        totalNorm += grad[i] * grad[i];
      }
    }
    totalNorm = Math.sqrt(totalNorm);

    
    const clipCoeff = this.maxGradNorm / (totalNorm + 1e-8);
    const shouldClip = totalNorm > this.maxGradNorm;

    for (const parameter of this.parameters) {
      if (!parameter._grad) continue;
      const grad = parameter._grad.data;
      const paramData = parameter.data;
      
      if (shouldClip) {
        
        for (let i = 0; i < grad.length; i++) {
          grad[i] *= clipCoeff;
        }
      }
      
      if (this.momentum > 0) {
        let velocity = this.velocities.get(parameter);
        if (!velocity) {
          velocity = new Float32Array(paramData.length);
          this.velocities.set(parameter, velocity);
        }

        
        for (let i = 0; i < paramData.length; i++) {
          velocity[i] = this.momentum * velocity[i] + grad[i];
          paramData[i] -= this.learningRate * velocity[i];
        }
      } else {
        
        for (let i = 0; i < paramData.length; i++) {
          paramData[i] -= this.learningRate * grad[i];
        }
      }
    }
  }
}




export * as plot from "./plot.ts"
export class Sequencial extends Module {
  modules: Module[] = [];
  constructor(modules: Module[]) {
    super()
    this.modules = modules
    this.register_parameters(this.modules)
  }

  override forward(_input: Tensor): Tensor {
    let res: Tensor = _input
    for (let i = 0; i < this.modules.length; i++) {
      res = this.modules[i].$(res)
    }
    return res
  }
}
export class Relu extends Module {
  constructor() {
    super()
    
  }

  override forward(_input: Tensor): Tensor {
    return relu(_input)
  }
}





export * as plot from "./plot.ts"